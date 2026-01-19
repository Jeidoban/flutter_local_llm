import 'dart:async';
import 'package:flutter/foundation.dart';
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
  bool _needsFullContextReset = true;

  /// Public access to chat history
  /// Users can read, modify, or replace this to control conversation context
  late ChatHistory chatHistory;

  // Private constructor
  FlutterLocalLlm._({
    required LLMIsolate isolate,
    required ChatFormat chatFormat,
    String? systemPrompt,
  }) : _isolate = isolate,
       _chatFormat = chatFormat {
    // Initialize chat history
    chatHistory = ChatHistory();

    // Add system prompt if provided
    if (systemPrompt != null && systemPrompt.isNotEmpty) {
      chatHistory.addMessage(role: Role.system, content: systemPrompt);
    }
  }

  /// Initialize FlutterLocalLlm with a model
  /// All parameters are optional - defaults to Gemma model with sensible settings
  ///
  /// Example usage:
  /// ```dart
  /// // Minimal - downloads and initializes Gemma model with defaults
  /// final llm = await FlutterLocalLlm.init();
  ///
  /// // With options
  /// final llm = await FlutterLocalLlm.init(
  ///   model: LLMModel.gemma3nE2B,
  ///   systemPrompt: 'You are a helpful assistant.',
  ///   onDownloadProgress: (progress) => print('${(progress * 100).toStringAsFixed(1)}%'),
  ///   contextSize: 16384,
  ///   nPredict: -1,
  ///   nBatch: 2048,
  ///   nThreads: 8,
  ///   temperature: 0.7,
  ///   topK: 64,
  ///   topP: 0.95,
  ///   minP: 0.05,
  ///   penaltyRepeat: 1.1,
  ///   keepRecentPairs: 2,
  /// );
  /// ```
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
    int keepRecentPairs = 2,
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
      keepRecentPairs: keepRecentPairs,
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

  /// Get remaining context space from the LLM
  Future<int> _getRemainingContextSpace() async {
    final requestId = _nextRequestId++;

    // Send command to isolate
    _isolate.sendCommand(
      GetRemainingContextCommand(requestId: requestId),
    );

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

  /// Clear chat history and start fresh conversation
  /// Keeps the same system prompt from init()
  /// Also clears the LLM context in the isolate
  void clearHistory() {
    if (kDebugMode) {
      print('[FlutterLocalLlm] Clearing chat history');
    }

    // Get current system prompt if any
    String? systemPrompt;
    if (chatHistory.messages.isNotEmpty &&
        chatHistory.messages.first.role == Role.system) {
      systemPrompt = chatHistory.messages.first.content;
    }

    // Create new chat history
    chatHistory = ChatHistory();

    // Re-add system prompt if it existed
    if (systemPrompt != null && systemPrompt.isNotEmpty) {
      chatHistory.addMessage(role: Role.system, content: systemPrompt);
    }

    // Clear the LLM context in the isolate
    _isolate.sendCommand(ClearContextCommand());

    // Mark that we need to send full context on next message
    _needsFullContextReset = true;
  }

  /// Send a message and get streaming tokens
  /// Automatically manages chat history
  ///
  /// Example usage:
  /// ```dart
  /// final stream = llm.sendMessage('Hello!', role: Role.user);
  /// await for (final token in stream) {
  ///   print(token);  // Print each token as it arrives
  /// }
  /// ```
  Stream<String> sendMessage(String message, {Role role = Role.user}) async* {
    // Create a temporary history with just this message
    final tempHistory = ChatHistory();
    tempHistory.addMessage(role: role, content: message);

    // Delegate to sendMessageWithHistory
    await for (final token in sendMessageWithHistory(tempHistory)) {
      yield token;
    }
  }

  /// Send a message and wait for the complete response
  /// Automatically manages chat history
  ///
  /// Example usage:
  /// ```dart
  /// final response = await llm.sendMessageComplete('Hello!', role: Role.user);
  /// print(response);  // Print the complete response
  /// ```
  Future<String> sendMessageComplete(
    String message, {
    Role role = Role.user,
  }) async {
    final buffer = StringBuffer();
    await for (final token in sendMessage(message, role: role)) {
      buffer.write(token);
    }
    return buffer.toString();
  }

  /// Send a ChatHistory with multiple messages and get streaming tokens
  /// This allows you to stack multiple messages before generating a response
  /// The provided messages will be added to the internal chat history
  ///
  /// Example usage:
  /// ```dart
  /// final history = ChatHistory();
  /// history.addMessage(role: Role.user, content: 'First message');
  /// history.addMessage(role: Role.user, content: 'Second message');
  /// history.addMessage(role: Role.user, content: 'Third message');
  ///
  /// final stream = llm.sendMessageWithHistory(history);
  /// await for (final token in stream) {
  ///   print(token);
  /// }
  /// ```
  Stream<String> sendMessageWithHistory(ChatHistory messages) async* {
    if (kDebugMode) {
      print(
        '[FlutterLocalLlm] Sending ChatHistory with ${messages.messages.length} messages',
      );
    }

    // Track how many messages we had before adding new ones
    final previousMessageCount = chatHistory.messages.length;

    // Add all messages from provided history to internal history
    for (final msg in messages.messages) {
      chatHistory.addMessage(role: msg.role, content: msg.content);
    }

    // Add empty assistant message to history
    chatHistory.addMessage(role: Role.assistant, content: '');

    // Check if we need to trim before generating
    // Get the messages we're about to send as a string to estimate token count
    final messagesToSend = StringBuffer();
    for (int i = previousMessageCount; i < chatHistory.messages.length; i++) {
      messagesToSend.write(chatHistory.messages[i].content);
    }

    // Check remaining context space
    final remaining = await _getRemainingContextSpace();

    // Rough estimate: 1 token ~= 4 characters, reserve extra space for response
    final estimatedTokensNeeded = (messagesToSend.length / 4).ceil() + 500;

    if (kDebugMode) {
      print('[FlutterLocalLlm] Remaining context: $remaining, estimated needed: $estimatedTokensNeeded');
    }

    // If we're running low on space, trim the history
    if (remaining < estimatedTokensNeeded) {
      if (kDebugMode) {
        print('[FlutterLocalLlm] Low context space! Auto-trimming history...');
      }

      // Remove the assistant message we just added temporarily
      chatHistory.messages.removeLast();

      // Trim the history
      _trimHistory();

      // Clear llama context
      _isolate.sendCommand(ClearContextCommand());

      // Re-add the assistant message
      chatHistory.addMessage(role: Role.assistant, content: '');

      // Mark that we need to send full context after trimming
      _needsFullContextReset = true;
    }

    // Generate prompt - either full context or just new messages
    final String prompt;
    if (_needsFullContextReset) {
      // First message, after clear, or after trimming - send full context
      if (kDebugMode) {
        print(
          '[FlutterLocalLlm] Sending full context (${chatHistory.messages.length} messages)',
        );
      }
      prompt = chatHistory.exportFormat(
        _chatFormat,
        leaveLastAssistantOpen: true,
      );
      _needsFullContextReset = false;
    } else {
      // Subsequent messages - only send new content
      if (kDebugMode) {
        print(
          '[FlutterLocalLlm] Sending incremental update (${chatHistory.messages.length - previousMessageCount} new messages)',
        );
      }

      // Create a temporary history with just the new messages
      final incrementalHistory = ChatHistory();
      for (int i = previousMessageCount; i < chatHistory.messages.length; i++) {
        incrementalHistory.addMessage(
          role: chatHistory.messages[i].role,
          content: chatHistory.messages[i].content,
        );
      }

      prompt = incrementalHistory.exportFormat(
        _chatFormat,
        leaveLastAssistantOpen: true,
      );
    }

    // Stream response
    final responseBuffer = StringBuffer();
    await for (final token in _generateFromPrompt(prompt)) {
      responseBuffer.write(token);
      yield token;
    }

    // Update the last assistant message with the complete response
    final fullResponse = responseBuffer.toString().trim();
    if (chatHistory.messages.isNotEmpty &&
        chatHistory.messages.last.role == Role.assistant) {
      chatHistory.messages.last = Message(
        role: Role.assistant,
        content: fullResponse,
      );
    }

    if (kDebugMode) {
      print(
        '[FlutterLocalLlm] ChatHistory response completed, response length: ${fullResponse.length}',
      );
    }
  }

  /// Trim chat history to keep only recent conversation
  void _trimHistory() {
    // Extract system prompt if present
    String? systemPrompt;
    if (chatHistory.messages.isNotEmpty &&
        chatHistory.messages.first.role == Role.system) {
      systemPrompt = chatHistory.messages.first.content;
    }

    // Get the keepRecentPairs value from config (default 2)
    final keepPairs = 2; // We can expose this later if needed

    // Keep only the most recent pairs (user + assistant)
    final recentMessages = <Message>[];
    int pairCount = 0;

    // Iterate backwards, collecting recent pairs
    for (int i = chatHistory.messages.length - 1; i >= 0; i--) {
      final msg = chatHistory.messages[i];

      // Skip system messages in the count
      if (msg.role == Role.system) continue;

      recentMessages.insert(0, msg);

      // Count complete pairs (assistant messages indicate end of pair)
      if (msg.role == Role.assistant && msg.content.isNotEmpty) {
        pairCount++;
        if (pairCount >= keepPairs) break;
      }
    }

    // Rebuild chat history
    chatHistory = ChatHistory();

    // Re-add system prompt
    if (systemPrompt != null && systemPrompt.isNotEmpty) {
      chatHistory.addMessage(role: Role.system, content: systemPrompt);
    }

    // Add recent messages
    for (final msg in recentMessages) {
      chatHistory.addMessage(role: msg.role, content: msg.content);
    }

    if (kDebugMode) {
      print('[FlutterLocalLlm] Trimmed to ${chatHistory.messages.length} messages');
    }
  }

  /// Send a ChatHistory with multiple messages and wait for the complete response
  /// This allows you to stack multiple messages before generating a response
  /// The provided messages will be added to the internal chat history
  ///
  /// Example usage:
  /// ```dart
  /// final history = ChatHistory();
  /// history.addMessage(role: Role.user, content: 'First message');
  /// history.addMessage(role: Role.user, content: 'Second message');
  ///
  /// final response = await llm.sendMessageWithHistoryComplete(history);
  /// print(response);  // Print the complete response
  /// ```
  Future<String> sendMessageWithHistoryComplete(ChatHistory messages) async {
    final buffer = StringBuffer();
    await for (final token in sendMessageWithHistory(messages)) {
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
