import 'dart:async';
import 'dart:isolate';
import 'package:flutter/foundation.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'models.dart';
import 'library_setup.dart';

// ============================================================================
// Commands sent to isolate
// ============================================================================

sealed class IsolateCommand {}

class InitializeCommand extends IsolateCommand {
  final String modelPath;
  final LLMConfig config;

  InitializeCommand({required this.modelPath, required this.config});
}

class GenerateFromPromptCommand extends IsolateCommand {
  final String prompt;
  final int requestId;

  GenerateFromPromptCommand({required this.prompt, required this.requestId});
}

class ClearContextCommand extends IsolateCommand {}

class DisposeCommand extends IsolateCommand {}

// ============================================================================
// Responses from isolate
// ============================================================================

sealed class IsolateResponse {}

class InitializedResponse extends IsolateResponse {}

class TokenResponse extends IsolateResponse {
  final String token;
  final int requestId;

  TokenResponse({required this.token, required this.requestId});
}

class CompletionResponse extends IsolateResponse {
  final int requestId;

  CompletionResponse({required this.requestId});
}

class ErrorResponse extends IsolateResponse {
  final String error;
  final int? requestId;

  ErrorResponse({required this.error, this.requestId});
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get stop tokens for a given chat format
List<String> _getStopTokens(ChatFormat chatFormat) {
  switch (chatFormat) {
    case ChatFormat.gemma:
      return ['<end_of_turn>', '<eos>'];
    case ChatFormat.chatml:
      return ['<|im_end|>'];
    case ChatFormat.alpaca:
      return ['### Response:', '### Instruction:'];
    default:
      // Default fallback for other formats
      return ['</s>', '<eos>'];
  }
}

/// Generate text from a prompt and stream tokens back to main isolate
Future<String> _generateFromPrompt(
  Llama llama,
  String prompt,
  ChatFormat chatFormat,
  int requestId,
  SendPort mainSendPort,
) async {
  if (kDebugMode) {
    print('[Isolate] Generating from prompt (ID: $requestId)');
    print('[Isolate] Prompt length: ${prompt.length}');
  }

  // Set the prompt
  llama.setPrompt(prompt);

  // Get stop tokens for this chat format
  final stopTokens = _getStopTokens(chatFormat);

  // Stream tokens and check for stop tokens
  final responseBuffer = StringBuffer();
  await for (final token in llama.generateText()) {
    // Check if this token is a stop token
    bool shouldStop = false;
    for (final stopToken in stopTokens) {
      if (responseBuffer.toString().endsWith(stopToken) ||
          token.contains(stopToken)) {
        shouldStop = true;
        // Remove the stop token from the response if it's already added
        final currentResponse = responseBuffer.toString();
        if (currentResponse.endsWith(stopToken)) {
          responseBuffer.clear();
          final endIndex = currentResponse.length - stopToken.length;
          responseBuffer.write(currentResponse.substring(0, endIndex));
        }
        break;
      }
    }

    if (shouldStop) {
      if (kDebugMode) {
        print('[Isolate] Hit stop token, stopping generation');
      }
      break;
    }

    responseBuffer.write(token);
    mainSendPort.send(TokenResponse(token: token, requestId: requestId));
  }

  // Get the full response and trim extra whitespace/newlines
  final fullResponse = responseBuffer.toString().trim();

  if (kDebugMode) {
    print(
      '[Isolate] Completed response (ID: $requestId), length: ${fullResponse.length}',
    );
  }

  mainSendPort.send(CompletionResponse(requestId: requestId));
  return fullResponse;
}

// ============================================================================
// Isolate Entry Point
// ============================================================================

void _isolateEntryPoint(SendPort mainSendPort) {
  final receivePort = ReceivePort();
  mainSendPort.send(receivePort.sendPort);

  Llama? llama;
  ChatFormat? chatFormat;

  receivePort.listen((message) async {
    try {
      if (message is InitializeCommand) {
        if (kDebugMode) {
          print('[Isolate] Initializing with model: ${message.modelPath}');
        }

        // Setup library path
        setupLlamaLibraryPath();

        // Create context params
        final contextParams = ContextParams();
        if (message.config.nPredict != null) {
          contextParams.nPredict = message.config.nPredict!;
        }
        if (message.config.contextSize != null) {
          contextParams.nCtx = message.config.contextSize!;
        }
        if (message.config.nBatch != null) {
          contextParams.nBatch = message.config.nBatch!;
        }

        // Create sampler params
        final samplerParams = SamplerParams();
        if (message.config.temperature != null) {
          samplerParams.temp = message.config.temperature!;
        }
        if (message.config.topK != null) {
          samplerParams.topK = message.config.topK!;
        }
        if (message.config.topP != null) {
          samplerParams.topP = message.config.topP!;
        }
        if (message.config.minP != null) {
          samplerParams.minP = message.config.minP!;
        }
        if (message.config.penaltyRepeat != null) {
          samplerParams.penaltyRepeat = message.config.penaltyRepeat!;
        }

        // Create Llama instance
        llama = Llama(
          message.modelPath,
          contextParams: contextParams,
          samplerParams: samplerParams,
        );

        // Store chat format for later use
        chatFormat = message.config.chatFormat;

        if (kDebugMode) {
          print('[Isolate] Initialization complete');
        }

        mainSendPort.send(InitializedResponse());
      } else if (message is GenerateFromPromptCommand) {
        if (llama == null || chatFormat == null) {
          mainSendPort.send(
            ErrorResponse(
              error: 'Llama not initialized',
              requestId: message.requestId,
            ),
          );
          return;
        }

        // Generate from the provided prompt
        await _generateFromPrompt(
          llama!,
          message.prompt,
          chatFormat!,
          message.requestId,
          mainSendPort,
        );
      } else if (message is ClearContextCommand) {
        if (llama != null) {
          if (kDebugMode) {
            print('[Isolate] Clearing context');
          }
          llama!.clear();
        }
      } else if (message is DisposeCommand) {
        if (kDebugMode) {
          print('[Isolate] Disposing resources');
        }

        llama?.dispose();
        llama = null;
        chatFormat = null;
      }
    } catch (e, stackTrace) {
      if (kDebugMode) {
        print('[Isolate] Error: $e');
        print('[Isolate] Stack trace: $stackTrace');
      }

      mainSendPort.send(
        ErrorResponse(
          error: e.toString(),
          requestId: message is GenerateFromPromptCommand
              ? message.requestId
              : null,
        ),
      );
    }
  });
}

// ============================================================================
// Isolate Manager
// ============================================================================

class LLMIsolate {
  final Isolate _isolate;
  final SendPort _sendPort;
  final ReceivePort _receivePort;
  final StreamController<IsolateResponse> _responseController;

  LLMIsolate._({
    required Isolate isolate,
    required SendPort sendPort,
    required ReceivePort receivePort,
    required Stream<dynamic> broadcastStream,
  }) : _isolate = isolate,
       _sendPort = sendPort,
       _receivePort = receivePort,
       _responseController = StreamController<IsolateResponse>.broadcast() {
    // Listen to all isolate responses and forward to stream controller
    broadcastStream.listen((message) {
      if (message is IsolateResponse) {
        _responseController.add(message);
      }
    });
  }

  /// Spawn a new isolate and initialize it with the model
  static Future<LLMIsolate> spawn(String modelPath, LLMConfig config) async {
    if (kDebugMode) {
      print('[LLMIsolate] Spawning isolate...');
    }

    // Create receive port for main isolate
    final receivePort = ReceivePort();

    // Convert to broadcast stream so we can listen multiple times
    final broadcastStream = receivePort.asBroadcastStream();

    // Spawn isolate
    final isolate = await Isolate.spawn(
      _isolateEntryPoint,
      receivePort.sendPort,
    );

    // Get send port from spawned isolate (first message)
    final sendPort = await broadcastStream.first as SendPort;

    if (kDebugMode) {
      print('[LLMIsolate] Isolate spawned, waiting for initialization...');
    }

    // Create isolate manager with broadcast stream
    final llmIsolate = LLMIsolate._(
      isolate: isolate,
      sendPort: sendPort,
      receivePort: receivePort,
      broadcastStream: broadcastStream,
    );

    // Send initialization command
    llmIsolate.sendCommand(
      InitializeCommand(modelPath: modelPath, config: config),
    );

    // Wait for initialization to complete
    await llmIsolate.responseStream.firstWhere(
      (response) => response is InitializedResponse,
    );

    if (kDebugMode) {
      print('[LLMIsolate] Isolate initialized successfully');
    }

    return llmIsolate;
  }

  /// Get stream of responses from isolate
  Stream<IsolateResponse> get responseStream => _responseController.stream;

  /// Send a command to the isolate
  void sendCommand(IsolateCommand command) {
    _sendPort.send(command);
  }

  /// Dispose the isolate and clean up resources
  void dispose() {
    if (kDebugMode) {
      print('[LLMIsolate] Disposing isolate...');
    }

    sendCommand(DisposeCommand());
    _isolate.kill();
    _receivePort.close();
    _responseController.close();

    if (kDebugMode) {
      print('[LLMIsolate] Isolate disposed');
    }
  }
}
