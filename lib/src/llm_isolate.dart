import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'models.dart';

// ============================================================================
// Commands sent to isolate
// ============================================================================

sealed class IsolateCommand {}

class InitializeCommand extends IsolateCommand {
  final String modelPath;
  final LLMConfig config;
  final String? imageModelPath;

  InitializeCommand({
    required this.modelPath,
    required this.config,
    this.imageModelPath,
  });
}

class GenerateFromPromptCommand extends IsolateCommand {
  final String prompt;
  final int requestId;
  final List<String>? attachmentPaths;

  GenerateFromPromptCommand({
    required this.prompt,
    required this.requestId,
    this.attachmentPaths,
  });
}

class ClearContextCommand extends IsolateCommand {}

class GetRemainingContextCommand extends IsolateCommand {
  final int requestId;

  GetRemainingContextCommand({required this.requestId});
}

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

class RemainingContextResponse extends IsolateResponse {
  final int remaining;
  final int requestId;

  RemainingContextResponse({required this.remaining, required this.requestId});
}

// ============================================================================
// Llama Manager
// ============================================================================

/// Manages the Llama instance and handles commands within the isolate
class LlamaManager {
  final SendPort _mainSendPort;
  late Llama _llama;
  late LLMConfig _config;

  LlamaManager(this._mainSendPort);

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
        return ['</s>', '<eos>'];
    }
  }

  /// Generate text from a prompt and stream tokens back to main isolate
  Future<void> _generateFromPrompt(
    String prompt,
    int requestId, {
    List<String>? attachmentPaths,
  }) async {
    final stopTokens = _getStopTokens(_config.chatFormat);

    Stream<String> stream;
    if (attachmentPaths != null && attachmentPaths.isNotEmpty) {
      // Use multimodal generation with attachments
      final attachments = attachmentPaths
          .map((path) => LlamaImage.fromFile(File(path)))
          .toList();
      stream = _llama.generateWithMedia(prompt, inputs: attachments);
    } else {
      // Use text-only generation
      _llama.setPrompt(prompt);
      stream = _llama.generateText();
    }

    await for (final token in stream) {
      bool shouldStop = false;
      for (final stopToken in stopTokens) {
        if (token.contains(stopToken)) {
          shouldStop = true;
          break;
        }
      }

      if (shouldStop) break;

      _mainSendPort.send(TokenResponse(token: token, requestId: requestId));
    }

    _mainSendPort.send(CompletionResponse(requestId: requestId));
  }

  void _setupLlamaLibraryPath() {
    // For iOS/macOS, the framework is embedded already and loaded
    // from process.

    if (Platform.isAndroid) {
      // For Android, set the path to the .so file
      Llama.libraryPath = 'libllama.so';
    } else if (Platform.isLinux) {
      Llama.libraryPath = 'libllama.so';
    } else if (Platform.isWindows) {
      Llama.libraryPath = 'llama.dll';
    }
  }

  /// Handle a command from the main isolate
  Future<void> handleCommand(IsolateCommand message) async {
    try {
      switch (message) {
        case InitializeCommand():
          _setupLlamaLibraryPath();

          _config = message.config;

          final contextParams = ContextParams()
            ..nPredict = _config.nPredict
            ..nCtx = _config.contextSize
            ..nBatch = _config.nBatch
            ..nThreads = _config.nThreads;

          final samplerParams = SamplerParams()
            ..temp = _config.temperature
            ..topK = _config.topK
            ..topP = _config.topP
            ..minP = _config.minP
            ..penaltyRepeat = _config.penaltyRepeat;

          _llama = Llama(
            message.modelPath,
            contextParams: contextParams,
            samplerParams: samplerParams,
            mmprojPath: message.imageModelPath,
          );

          _mainSendPort.send(InitializedResponse());
        case GenerateFromPromptCommand():
          await _generateFromPrompt(
            message.prompt,
            message.requestId,
            attachmentPaths: message.attachmentPaths,
          );
        case GetRemainingContextCommand():
          final remaining = _llama.getRemainingContextSpace();
          _mainSendPort.send(
            RemainingContextResponse(
              remaining: remaining,
              requestId: message.requestId,
            ),
          );
        case ClearContextCommand():
          _llama.clear();
        case DisposeCommand():
          dispose();
      }
    } catch (e) {
      _mainSendPort.send(
        ErrorResponse(
          error: e.toString(),
          requestId: message is GenerateFromPromptCommand
              ? message.requestId
              : null,
        ),
      );
    }
  }

  /// Clean up resources
  void dispose() {
    _llama.dispose();
  }
}

// ============================================================================
// Isolate Entry Point
// ============================================================================

void _isolateEntryPoint(SendPort mainSendPort) {
  final receivePort = ReceivePort();
  mainSendPort.send(receivePort.sendPort);

  final manager = LlamaManager(mainSendPort);

  receivePort.listen((message) async {
    await manager.handleCommand(message);
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
  static Future<LLMIsolate> spawn(
    String modelPath,
    LLMConfig config, {
    String? imageModelPath,
  }) async {
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

    // Create isolate manager with broadcast stream
    final llmIsolate = LLMIsolate._(
      isolate: isolate,
      sendPort: sendPort,
      receivePort: receivePort,
      broadcastStream: broadcastStream,
    );

    // Send initialization command
    llmIsolate.sendCommand(
      InitializeCommand(
        modelPath: modelPath,
        config: config,
        imageModelPath: imageModelPath,
      ),
    );

    // Wait for initialization to complete
    final response = await llmIsolate.responseStream.firstWhere(
      (response) =>
          response is InitializedResponse || response is ErrorResponse,
    );

    if (response is ErrorResponse) {
      llmIsolate.dispose();
      throw Exception('Failed to initialize model: ${response.error}');
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
    sendCommand(DisposeCommand());
    _isolate.kill();
    _receivePort.close();
    _responseController.close();
  }
}
