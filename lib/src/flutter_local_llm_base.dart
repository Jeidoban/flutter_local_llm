import 'dart:async';
import 'dart:io';
import 'package:flutter_local_llm/src/llm_chat_history.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'models.dart';
import 'llm_isolate.dart';
import 'model_manager.dart';
import 'chat_manager.dart';

/// Main class for running local LLMs on device with automatic model downloading
/// and context management.
///
/// ```dart
/// final llm = await FlutterLocalLlm.init();
/// await for (final token in llm.sendMessage('Hello!')) {
///   print(token);
/// }
/// llm.dispose();
/// ```
class FlutterLocalLlm {
  final LlmIsolate _isolate;
  final LlmConfig config;
  final ChatManager chatManager;
  final ModelManager modelManager;
  int _nextRequestId = 0;

  // Private constructor - accepts all dependencies
  FlutterLocalLlm._({
    required LlmIsolate isolate,
    required this.config,
    required this.chatManager,
    required this.modelManager,
  }) : _isolate = isolate;

  /// Simple factory with defaults - zero config option
  static Future<FlutterLocalLlm> create({
    LlmModel model = LlmModel.gemma3_1b_q5,
    String? systemPrompt,
    void Function(double progress)? onDownloadProgress,
    int contextSize = 8096,
    int nPredict = -1,
    int? nBatch,
    int nThreads = 8,
    double temperature = 0.7,
    int topK = 64,
    double topP = 0.95,
    double minP = 0.05,
    double penaltyRepeat = 1.1,
    // Path customization
    String? modelsPath,
    String? chatStoragePath,
  }) {
    // Default batch size to context size if not specified
    final batchSize = nBatch ?? contextSize;

    // Build system prompt with context size guidance if not provided
    final effectiveSystemPrompt =
        systemPrompt ??
        'You are a helpful assistant. Please keep your responses concise, '
            'answer directly what the user asks, avoid unnecessary elaboration. '
            'Limit answers to an absolute maximum of ${(contextSize / 2).round()} '
            'tokens to fit within the conversation context.';

    // Map enum to base config, then apply overrides
    final baseConfig = switch (model) {
      LlmModel.gemma3_1b_q5 => LlmConfig.gemma3_1b_q5,
      LlmModel.gemma3n_E2B_q4 => LlmConfig.gemma3n_E2B_q4,
      LlmModel.gemma3_4b_q5_mm => LlmConfig.gemma3_4b_q5_mm,
      LlmModel.gemma3_4b_q3_mm => LlmConfig.gemma3_4b_q3_mm,
    };

    final config = baseConfig.copyWith(
      systemPrompt: effectiveSystemPrompt,
      contextSize: contextSize,
      nPredict: nPredict,
      nBatch: batchSize,
      nThreads: nThreads,
      temperature: temperature,
      topK: topK,
      topP: topP,
      minP: minP,
      penaltyRepeat: penaltyRepeat,
    );

    // For multimodal models, weight progress: text model 0-50%, image model 50-100%
    final hasImageModel =
        config.imageUrl != null && config.imageFileName != null;
    int downloadCount = 0;

    final modelManager = ModelManager(
      modelsPath: modelsPath,
      onDownloadProgress: onDownloadProgress == null
          ? null
          : (progress) {
              final weighted = hasImageModel
                  ? (downloadCount == 0 ? progress * 0.5 : 0.5 + progress * 0.5)
                  : progress;
              onDownloadProgress(weighted);
              if (progress >= 1.0) downloadCount++;
            },
    );

    // Calculate keepRecentPairs based on context size
    final keepRecentPairs = (contextSize / 2048).round().clamp(1, 10);

    return createCustom(
      config: config,
      modelManager: modelManager,
      chatManager: ChatManager(
        storagePath: chatStoragePath,
        keepRecentPairs: keepRecentPairs,
      ),
    );
  }

  /// Factory with optional dependency injection — for testing or custom implementations.
  ///
  /// Pass a custom [modelManager] or [chatManager] to override the defaults.
  /// If omitted, defaults are created from [config].
  static Future<FlutterLocalLlm> createCustom({
    required LlmConfig config,
    ModelManager? modelManager,
    ChatManager? chatManager,
  }) async {
    final effectiveModelManager = modelManager ?? ModelManager();
    final keepRecentPairs = (config.contextSize / 2048).round().clamp(1, 10);
    final effectiveChatManager =
        chatManager ?? ChatManager(keepRecentPairs: keepRecentPairs);

    // Create isolate (downloads models if needed)
    final isolate = await effectiveModelManager.createModelIsolate(config);

    // Set up context clearing callback
    effectiveChatManager.onSessionChanged = () =>
        isolate.sendCommand(ClearContextCommand());

    // Create instance
    final instance = FlutterLocalLlm._(
      isolate: isolate,
      config: config,
      chatManager: effectiveChatManager,
      modelManager: effectiveModelManager,
    );

    // Load chats from storage
    await effectiveChatManager.loadChats();

    // Ensure there's always at least one chat with the right system prompt
    if (effectiveChatManager.chats.isEmpty) {
      effectiveChatManager.startNewChat(systemPrompt: config.systemPrompt);
    }

    return instance;
  }

  /// Internal helper to generate from a prompt
  Stream<String> _generateFromPrompt(
    String prompt, {
    List<String>? filePaths,
  }) async* {
    final requestId = _nextRequestId++;

    // Send command to isolate
    _isolate.sendCommand(
      GenerateFromPromptCommand(
        prompt: prompt,
        requestId: requestId,
        attachmentPaths: filePaths,
      ),
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

  /// Clear the current chat's history and LLM context
  Future<void> clearHistory() async {
    chatManager.activeChat.messages.clear();
    chatManager.activeChat.fullHistory.clear();

    if (config.systemPrompt.isNotEmpty) {
      chatManager.activeChat.addMessage(
        role: Role.system,
        content: config.systemPrompt,
      );
    }

    _isolate.sendCommand(ClearContextCommand());

    // Save to storage
    await chatManager.saveChats();
  }

  /// Send a message and get streaming tokens
  ///
  /// By default, adds the message and response to chat history.
  /// Set [addToHistory] to false for stateless generation.
  /// Optionally attach [images] for multimodal input.
  Stream<String> sendMessage(
    String message, {
    Role role = Role.user,
    bool addToHistory = true,
    List<File>? images,
  }) async* {
    final tempHistory = LlmChatHistory();
    tempHistory.addMessage(
      role: role,
      content: message,
      images: images?.map((file) => file.path).toList(),
    );

    yield* sendMessageWithHistory(tempHistory, addToHistory: addToHistory);
  }

  /// Send a message and wait for complete response
  /// Optionally attach [images] for multimodal input.
  Future<String> sendMessageComplete(
    String message, {
    Role role = Role.user,
    bool addToHistory = true,
    List<File>? images,
  }) async {
    final buffer = StringBuffer();
    await for (final token in sendMessage(
      message,
      role: role,
      addToHistory: addToHistory,
      images: images,
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
      config.chatFormat,
      leaveLastAssistantOpen: true,
    );

    // Check if context is empty (just cleared or first message)
    final contextIsEmpty = remainingSpace >= config.contextSize - 100;

    // Count images in new messages for token estimation
    final newMessageImageCount = messages.messages
        .expand((msg) => msg.images)
        .length;

    // Determine if we need to rebuild context with history
    final needsHistoryRebuild =
        contextIsEmpty ||
        chatManager.activeChat.shouldTrimBeforePromptNoLlama(
          newPrompt,
          remainingSpace,
          config.contextSize,
          imageCount: newMessageImageCount,
        );

    if (needsHistoryRebuild) {
      // Create test prompt with full history to check if it fits
      final testMessages = LlmChatHistory();
      testMessages.messages.addAll(chatManager.activeChat.messages);
      testMessages.messages.addAll(messages.messages);
      final fullPrompt = testMessages.exportFormat(
        config.chatFormat,
        leaveLastAssistantOpen: true,
      );

      // Count total images (history + new messages)
      final totalImageCount = testMessages.messages
          .expand((msg) => msg.images)
          .length;

      // Trim history if full prompt doesn't fit
      if (chatManager.activeChat.shouldTrimBeforePromptNoLlama(
        fullPrompt,
        remainingSpace,
        config.contextSize,
        imageCount: totalImageCount,
      )) {
        chatManager.activeChat.autoTrimForSpaceNoLlama(
          keepRecentPairs: chatManager.keepRecentPairs,
        );
      }

      // Clear context and add (possibly trimmed) history
      _isolate.sendCommand(ClearContextCommand());
      tempMessages.messages.addAll(chatManager.activeChat.messages);
    }

    tempMessages.messages.addAll(messages.messages);

    var imagePaths = tempMessages.messages
        .expand((message) => message.images)
        .toList();

    final promptToSend = tempMessages.exportFormat(
      config.chatFormat,
      leaveLastAssistantOpen: true,
    );

    final responseBuffer = StringBuffer();
    await for (final token in _generateFromPrompt(
      promptToSend,
      filePaths: imagePaths,
    )) {
      responseBuffer.write(token);
      yield token;
    }

    if (addToHistory) {
      // Find first user message for auto-titling
      String? firstUserMessage;
      for (final msg in messages.messages) {
        if (msg.role == Role.user) {
          firstUserMessage = msg.content;
          break;
        }
      }

      // Auto-title from first user message if still "New Chat"
      if (chatManager.activeChat.title == 'New Chat' &&
          firstUserMessage != null) {
        chatManager.activeChat.title = firstUserMessage.length > 40
            ? '${firstUserMessage.substring(0, 40)}...'
            : firstUserMessage;
      }

      for (final msg in messages.messages) {
        chatManager.activeChat.addMessage(
          role: msg.role,
          content: msg.content,
          images: msg.images,
        );
      }

      chatManager.activeChat.addMessage(
        role: Role.assistant,
        content: responseBuffer.toString().trim(),
      );

      // Update timestamp and save
      chatManager.activeChat.updatedAt = DateTime.now();
      await chatManager.saveChats();
    }
  }

  /// Send multiple messages and wait for complete response
  /// Optionally attach [images] for multimodal input.
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
