import 'dart:async';
import 'dart:io';
import 'package:flutter_local_llm/src/llm_chat_history.dart';
import 'package:http/http.dart' as http;
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'models.dart';
import 'llm_isolate.dart';

/// Main class for running local LLMs on device with automatic model downloading
/// and context management.
///
/// ```dart
/// final llm = await FlutterLocalLlm.init(
///   model: LLMModel.gemma3nE2B,
///   systemPrompt: 'You are a helpful assistant.',
/// );
///
/// await for (final token in llm.sendMessage('Hello!')) {
///   print(token);
/// }
///
/// llm.dispose();
/// ```
class FlutterLocalLlm {
  final LLMIsolate _isolate;
  final LLMConfig _config;
  int _nextRequestId = 0;

  /// Public access to chat history
  /// Users can read, modify, or replace this to control conversation context
  late LlmChatHistory chatHistory;

  /// Get the current LLM configuration
  LLMConfig get config => _config;

  // Private constructor
  FlutterLocalLlm._({required LLMIsolate isolate, required LLMConfig config})
    : _isolate = isolate,
      _config = config {
    // Initialize chat history
    chatHistory = LlmChatHistory();

    // Add system prompt if provided
    if (config.systemPrompt != null && config.systemPrompt!.isNotEmpty) {
      chatHistory.addMessage(role: Role.system, content: config.systemPrompt!);
    }
  }

  /// Initialize FlutterLocalLlm with a model
  static Future<FlutterLocalLlm> init({
    LLMModel model = LLMModel.gemma3nE2B,
    String? systemPrompt,
    String? customUrl,
    void Function(double progress)? onDownloadProgress,
    int contextSize = 16384,
    int nPredict = -1,
    int nBatch = 2048,
    int nThreads = 8,
    double temperature = 0.7,
    int topK = 64,
    double topP = 0.95,
    double minP = 0.05,
    double penaltyRepeat = 1.1,
  }) async {
    // Create config
    final config = LLMConfig(
      model: model,
      customUrl: customUrl,
      systemPrompt: systemPrompt ?? 'You are a helpful assistant.',
      contextSize: contextSize,
      nPredict: nPredict,
      nBatch: nBatch,
      nThreads: nThreads,
      temperature: temperature,
      topK: topK,
      topP: topP,
      minP: minP,
      penaltyRepeat: penaltyRepeat,
    );

    // Download model if needed
    final modelPath = await _getModelPath(
      config.downloadUrl,
      config.fileName,
      onDownloadProgress,
    );

    // Spawn isolate and initialize
    final isolate = await LLMIsolate.spawn(modelPath, config);

    return FlutterLocalLlm._(isolate: isolate, config: config);
  }

  /// Get the models directory, creating it if it doesn't exist
  static Future<Directory> _getModelsDirectory() async {
    final documentsDir = await getApplicationDocumentsDirectory();
    final modelsDir = Directory(
      path.join(documentsDir.path, 'flutter_local_llm_models'),
    );

    if (!modelsDir.existsSync()) {
      modelsDir.createSync(recursive: true);
    }

    return modelsDir;
  }

  /// Download a model file from a URL
  static Future<void> _downloadModel(
    String url,
    String destinationPath,
    void Function(double progress)? onProgress,
  ) async {
    final request = await http.Client().send(
      http.Request('GET', Uri.parse(url)),
    );
    final totalBytes = request.contentLength ?? 0;
    int downloadedBytes = 0;

    final file = File(destinationPath);
    final sink = file.openWrite();

    await for (final chunk in request.stream) {
      sink.add(chunk);
      downloadedBytes += chunk.length;

      if (onProgress != null && totalBytes > 0) {
        final progress = downloadedBytes / totalBytes;
        onProgress(progress);
      }
    }

    await sink.close();
  }

  /// Get the full path to a model file, downloading it if necessary
  static Future<String> _getModelPath(
    String downloadUrl,
    String fileName,
    void Function(double progress)? onDownloadProgress,
  ) async {
    final modelsDir = await _getModelsDirectory();
    final modelFilePath = path.join(modelsDir.path, fileName);

    if (!File(modelFilePath).existsSync()) {
      await _downloadModel(downloadUrl, modelFilePath, onDownloadProgress);
    }

    return modelFilePath;
  }

  /// Internal helper to generate from a prompt
  Stream<String> _generateFromPrompt(String prompt) async* {
    final requestId = _nextRequestId++;

    // Send command to isolate
    _isolate.sendCommand(
      GenerateFromPromptCommand(prompt: prompt, requestId: requestId),
    );

    // Listen for tokens
    await for (final response in _isolate.responseStream) {
      if (response is TokenResponse && response.requestId == requestId) {
        yield response.token;
      } else if (response is CompletionResponse &&
          response.requestId == requestId) {
        break;
      } else if (response is ErrorResponse) {
        throw Exception(response.error);
      }
    }
  }

  /// Get remaining context space from the isolate
  Future<int> getRemainingContextSpace() async {
    final requestId = _nextRequestId++;
    _isolate.sendCommand(GetRemainingContextCommand(requestId: requestId));

    // Wait for response
    await for (final response in _isolate.responseStream) {
      if (response is RemainingContextResponse &&
          response.requestId == requestId) {
        return response.remaining;
      } else if (response is ErrorResponse && response.requestId == requestId) {
        throw Exception(response.error);
      }
    }
    throw Exception('Failed to get remaining context space');
  }

  /// Clear chat history and LLM context
  void clearHistory() {
    chatHistory = LlmChatHistory();

    if (_config.systemPrompt != null && _config.systemPrompt!.isNotEmpty) {
      chatHistory.addMessage(role: Role.system, content: _config.systemPrompt!);
    }

    _isolate.sendCommand(ClearContextCommand());
  }

  /// Send a message and get streaming tokens
  ///
  /// By default, adds the message and response to chat history.
  /// Set [addToHistory] to false for stateless generation.
  Stream<String> sendMessage(
    String message, {
    Role role = Role.user,
    bool addToHistory = true,
  }) async* {
    final tempHistory = LlmChatHistory();
    tempHistory.addMessage(role: role, content: message);

    yield* sendMessageWithHistory(tempHistory, addToHistory: addToHistory);
  }

  /// Send a message and wait for complete response
  Future<String> sendMessageComplete(
    String message, {
    Role role = Role.user,
    bool addToHistory = true,
  }) async {
    final buffer = StringBuffer();
    await for (final token in sendMessage(
      message,
      role: role,
      addToHistory: addToHistory,
    )) {
      buffer.write(token);
    }
    return buffer.toString();
  }

  /// Send multiple messages and get streaming tokens
  ///
  /// By default, adds messages and response to chat history.
  /// Set [addToHistory] to false for stateless generation.
  Stream<String> sendMessageWithHistory(
    LlmChatHistory messages, {
    bool addToHistory = true,
  }) async* {
    final tempMessages = LlmChatHistory();
    int remainingSpace = await getRemainingContextSpace();
    String newPrompt = messages.exportFormat(
      _config.chatFormat,
      leaveLastAssistantOpen: true,
    );
    if (chatHistory.shouldTrimBeforePromptNoLlama(remainingSpace, newPrompt)) {
      chatHistory.autoTrimForSpaceNoLlama(remainingSpace);
      _isolate.sendCommand(ClearContextCommand());
      tempMessages.messages.addAll(chatHistory.messages);
    }

    tempMessages.messages.addAll(messages.messages);

    final promptToSend = tempMessages.exportFormat(
      _config.chatFormat,
      leaveLastAssistantOpen: true,
    );
    final responseBuffer = StringBuffer();
    await for (final token in _generateFromPrompt(promptToSend)) {
      responseBuffer.write(token);
      yield token;
    }

    if (addToHistory) {
      for (final msg in messages.messages) {
        chatHistory.addMessage(role: msg.role, content: msg.content);
      }

      chatHistory.addMessage(
        role: Role.assistant,
        content: responseBuffer.toString().trim(),
      );
    }
  }

  /// Send multiple messages and wait for complete response
  Future<String> sendMessageWithHistoryComplete(
    LlmChatHistory messages, {
    bool addToHistory = true,
  }) async {
    final buffer = StringBuffer();
    await for (final token in sendMessageWithHistory(
      messages,
      addToHistory: addToHistory,
    )) {
      buffer.write(token);
    }
    return buffer.toString();
  }

  /// Clean up resources
  void dispose() {
    _isolate.dispose();
  }
}
