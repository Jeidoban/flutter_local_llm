import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter_local_llm/src/llm_chat_history.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'models.dart';
import 'downloader.dart';
import 'llm_isolate.dart';

/// Main class for FlutterLocalLLM
/// Provides a batteries-included API for running local LLMs on device
class FlutterLocalLlm {
  final LLMIsolate _isolate;
  final ChatFormat _chatFormat;
  int _nextRequestId = 0;

  /// Public access to chat history
  /// Users can read, modify, or replace this to control conversation context
  late LlmChatHistory chatHistory;

  // Track system prompt separately
  final String? _systemPrompt;

  // Private constructor
  FlutterLocalLlm._({
    required LLMIsolate isolate,
    required ChatFormat chatFormat,
    required int contextSize,
    String? systemPrompt,
  }) : _isolate = isolate,
       _chatFormat = chatFormat,
       _systemPrompt = systemPrompt {
    // Initialize chat history
    chatHistory = LlmChatHistory();

    // Add system prompt if provided
    if (systemPrompt != null && systemPrompt.isNotEmpty) {
      chatHistory.addMessage(role: Role.system, content: systemPrompt);
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
    if (kDebugMode) {
      print('[FlutterLocalLlm.init] Starting initialization...');
    }

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

    if (kDebugMode) {
      print('[FlutterLocalLlm.init] Model: ${config.model.name}');
      print('[FlutterLocalLlm.init] Download URL: ${config.downloadUrl}');
    }

    // Download model if needed
    final modelPath = await getModelPath(
      config.downloadUrl,
      config.fileName,
      onDownloadProgress,
    );

    if (kDebugMode) {
      print('[FlutterLocalLlm.init] Model path: $modelPath');
      print('[FlutterLocalLlm.init] Spawning isolate...');
    }

    // Spawn isolate and initialize
    final isolate = await LLMIsolate.spawn(modelPath, config);

    if (kDebugMode) {
      print('[FlutterLocalLlm.init] Initialization complete!');
    }

    return FlutterLocalLlm._(
      isolate: isolate,
      chatFormat: config.chatFormat,
      contextSize: config.contextSize,
      systemPrompt: systemPrompt,
    );
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
        if (kDebugMode) {
          print('[FlutterLocalLlm] Error: ${response.error}');
        }
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
    // Create new history
    chatHistory = LlmChatHistory();

    // Re-add system prompt if it exists
    if (_systemPrompt != null && _systemPrompt.isNotEmpty) {
      chatHistory.addMessage(role: Role.system, content: _systemPrompt);
    }
    // Clear LLM context
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

    await for (final token in sendMessageWithHistory(
      tempHistory,
      addToHistory: addToHistory,
    )) {
      yield token;
    }
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
      _chatFormat,
      leaveLastAssistantOpen: true,
    );
    if (chatHistory.shouldTrimBeforePromptNoLlama(remainingSpace, newPrompt)) {
      chatHistory.autoTrimForSpaceNoLlama(remainingSpace);
      _isolate.sendCommand(ClearContextCommand());
      tempMessages.messages.addAll(chatHistory.messages);
    }

    tempMessages.messages.addAll(messages.messages);

    final promptToSend = tempMessages.exportFormat(
      _chatFormat,
      leaveLastAssistantOpen: true,
    );
    final responseBuffer = StringBuffer();
    await for (final token in _generateFromPrompt(promptToSend)) {
      responseBuffer.write(token);
      yield token;
    }

    if (addToHistory) {
      // Add new messages to history
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
    if (kDebugMode) {
      print('[FlutterLocalLlm] Disposing...');
    }
    _isolate.dispose();
  }
}
