import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import '../models/llm_config.dart';

// ============================================================================
// Commands sent to isolate
// ============================================================================

sealed class IsolateCommand {}

class InitializeCommand extends IsolateCommand {
  final String modelPath;
  final LlmConfig config;
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

/// Runs inside the spawned isolate and owns the [Llama] instance.
///
/// Receives [IsolateCommand] objects from the main isolate and sends
/// [IsolateResponse] objects back over [_mainSendPort]. Errors from any
/// command are caught and returned as [ErrorResponse] so the main isolate
/// can surface them to callers.
class LlamaManager {
  final SendPort _mainSendPort;
  late Llama _llama;
  late LlmConfig _config;

  LlamaManager(this._mainSendPort);

  /// Returns the stop tokens for the given [chatFormat].
  ///
  /// Generation stops when any of these tokens appear in the output stream.
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

  /// Runs inference on [prompt] and streams [TokenResponse] objects back to the main isolate.
  ///
  /// For multimodal prompts, loads each path in [attachmentPaths] as a
  /// [LlamaImage] and uses [Llama.generateWithMedia]. For text-only prompts,
  /// uses [Llama.generateText]. Sends a [CompletionResponse] when done.
  Future<void> _generateFromPrompt(
    String prompt,
    int requestId, {
    List<String>? attachmentPaths,
  }) async {
    final stopTokens = _getStopTokens(_config.chatFormat);

    Stream<String> stream;
    if (attachmentPaths != null && attachmentPaths.isNotEmpty) {
      final attachments = attachmentPaths
          .map((path) => LlamaImage.fromFile(File(path)))
          .toList();
      stream = _llama.generateWithMedia(prompt, inputs: attachments);
    } else {
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

  /// Sets the native library path based on the current platform.
  ///
  /// iOS and macOS load llama.cpp from the embedded framework automatically.
  /// Android, Linux, and Windows require an explicit path to the shared library.
  void _setupLlamaLibraryPath() {
    if (Platform.isAndroid) {
      Llama.libraryPath = 'libllama.so';
    } else if (Platform.isLinux) {
      Llama.libraryPath = 'libllama.so';
    } else if (Platform.isWindows) {
      Llama.libraryPath = 'llama.dll';
    }
  }

  /// Dispatches an [IsolateCommand] to the appropriate handler.
  ///
  /// Any uncaught exception is returned to the main isolate as an [ErrorResponse].
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

  /// Frees the native [Llama] resources.
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

/// Manages a spawned Dart isolate running [LlamaManager].
///
/// Exposes a command/response interface: send typed [IsolateCommand] objects
/// via [sendCommand] and receive typed [IsolateResponse] objects from
/// [responseStream]. The isolate is spawned and fully initialized before
/// [spawn] returns.
class LlmIsolate {
  final Isolate _isolate;
  final SendPort _sendPort;
  final ReceivePort _receivePort;
  final StreamController<IsolateResponse> _responseController;

  LlmIsolate._({
    required Isolate isolate,
    required SendPort sendPort,
    required ReceivePort receivePort,
    required Stream<dynamic> broadcastStream,
  }) : _isolate = isolate,
       _sendPort = sendPort,
       _receivePort = receivePort,
       _responseController = StreamController<IsolateResponse>.broadcast() {
    broadcastStream.listen((message) {
      if (message is IsolateResponse) {
        _responseController.add(message);
      }
    });
  }

  /// Spawns a new isolate, loads the model at [modelPath], and waits for it to initialize.
  ///
  /// Pass [imageModelPath] for multimodal models. Throws if initialization fails.
  static Future<LlmIsolate> spawn(
    String modelPath,
    LlmConfig config, {
    String? imageModelPath,
  }) async {
    final receivePort = ReceivePort();

    // We use a broadcast stream so we can listen multiple times.
    final broadcastStream = receivePort.asBroadcastStream();

    final isolate = await Isolate.spawn(
      _isolateEntryPoint,
      receivePort.sendPort,
    );

    final sendPort = await broadcastStream.first as SendPort;

    final llmIsolate = LlmIsolate._(
      isolate: isolate,
      sendPort: sendPort,
      receivePort: receivePort,
      broadcastStream: broadcastStream,
    );

    llmIsolate.sendCommand(
      InitializeCommand(
        modelPath: modelPath,
        config: config,
        imageModelPath: imageModelPath,
      ),
    );

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

  /// Broadcast stream of all responses from the isolate.
  Stream<IsolateResponse> get responseStream => _responseController.stream;

  /// Sends a command to the isolate.
  void sendCommand(IsolateCommand command) {
    _sendPort.send(command);
  }

  /// Kills the isolate and releases all associated resources.
  void dispose() {
    sendCommand(DisposeCommand());
    _isolate.kill();
    _receivePort.close();
    _responseController.close();
  }
}
