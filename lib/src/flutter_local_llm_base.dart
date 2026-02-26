import 'dart:async';
import 'dart:io';
import 'package:flutter_local_llm/src/models/llm_chat_history.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'models/llm_config.dart';
import 'isolate/llm_isolate.dart';
import 'managers/model_manager.dart';
import 'managers/chat_manager.dart';

/// Runs a local LLM on device with automatic model downloading and context management.
///
/// ```dart
/// final llm = await FlutterLocalLlm.create();
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

  FlutterLocalLlm._({
    required LlmIsolate isolate,
    required this.config,
    required this.chatManager,
    required this.modelManager,
  }) : _isolate = isolate;

  /// Creates a [FlutterLocalLlm] with a preset model and sensible defaults.
  ///
  /// For custom models or custom dependencies, use [createCustom].
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
    final batchSize = nBatch ?? contextSize;

    final effectiveSystemPrompt =
        systemPrompt ??
        'You are a helpful assistant. Please keep your responses concise, '
            'answer directly what the user asks, avoid unnecessary elaboration. '
            'Limit answers to an absolute maximum of ${(contextSize / 2).round()} '
            'tokens to fit within the conversation context.';

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

    // For multimodal models, weight progress: text model 0–50%, image model 50–100%
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

  /// Creates a [FlutterLocalLlm] with explicit dependencies.
  ///
  /// Provide a custom [ModelManager] or [ChatManager] to override default
  /// behavior — useful for testing or advanced configuration.
  static Future<FlutterLocalLlm> createCustom({
    required LlmConfig config,
    ModelManager? modelManager,
    ChatManager? chatManager,
  }) async {
    final effectiveModelManager = modelManager ?? ModelManager();
    final keepRecentPairs = (config.contextSize / 2048).round().clamp(1, 10);
    final effectiveChatManager =
        chatManager ?? ChatManager(keepRecentPairs: keepRecentPairs);

    final isolate = await effectiveModelManager.createModelIsolate(config);

    effectiveChatManager.onSessionChanged = () =>
        isolate.sendCommand(ClearContextCommand());

    final instance = FlutterLocalLlm._(
      isolate: isolate,
      config: config,
      chatManager: effectiveChatManager,
      modelManager: effectiveModelManager,
    );

    await effectiveChatManager.loadChats();

    if (effectiveChatManager.chats.isEmpty) {
      effectiveChatManager.startNewChat(systemPrompt: config.systemPrompt);
    }

    return instance;
  }

  /// Sends [prompt] to the isolate and yields tokens as they arrive.
  Stream<String> _generateFromPrompt(
    String prompt, {
    List<String>? filePaths,
  }) async* {
    final requestId = _nextRequestId++;

    _isolate.sendCommand(
      GenerateFromPromptCommand(
        prompt: prompt,
        requestId: requestId,
        attachmentPaths: filePaths,
      ),
    );

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

  /// Returns the number of tokens remaining in the model's context window.
  Future<int> getRemainingContextSpace() async {
    final requestId = _nextRequestId++;
    _isolate.sendCommand(GetRemainingContextCommand(requestId: requestId));

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

  /// Clears the active chat's message history and resets the LLM context.
  ///
  /// Re-adds the system prompt if one is configured, then persists the change.
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
    await chatManager.saveChats();
  }

  /// Sends a message and returns a stream of response tokens.
  ///
  /// Set [addToHistory] to false for stateless generation that does not
  /// affect the conversation history. Pass [images] for multimodal input.
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

  /// Sends a message and returns the complete response as a single string.
  ///
  /// Set [addToHistory] to false for stateless generation that does not
  /// affect the conversation history. Pass [images] for multimodal input.
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

  /// Sends a pre-built [LlmChatHistory] and returns a stream of response tokens.
  ///
  /// Set [addToHistory] to false for stateless generation that does not
  /// affect the conversation history. Attach images inside the history object
  /// for multimodal input.
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

    final newMessageImageCount = messages.messages
        .expand((msg) => msg.images)
        .length;

    final needsHistoryRebuild =
        contextIsEmpty ||
        chatManager.activeChat.shouldTrimBeforePromptNoLlama(
          newPrompt,
          remainingSpace,
          config.contextSize,
          imageCount: newMessageImageCount,
        );

    if (needsHistoryRebuild) {
      // Build a combined prompt to check if the full history fits
      final testMessages = LlmChatHistory();
      testMessages.messages.addAll(chatManager.activeChat.messages);
      testMessages.messages.addAll(messages.messages);
      final fullPrompt = testMessages.exportFormat(
        config.chatFormat,
        leaveLastAssistantOpen: true,
      );

      final totalImageCount = testMessages.messages
          .expand((msg) => msg.images)
          .length;

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

      // Clear context and repopulate with (possibly trimmed) history
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
      // Find the first user message to use as the auto-generated chat title
      String? firstUserMessage;
      for (final msg in messages.messages) {
        if (msg.role == Role.user) {
          firstUserMessage = msg.content;
          break;
        }
      }

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

      chatManager.activeChat.updatedAt = DateTime.now();
      await chatManager.saveChats();
    }
  }

  /// Sends a pre-built [LlmChatHistory] and returns the complete response as a single string.
  ///
  /// Set [addToHistory] to false for stateless generation that does not
  /// affect the conversation history. Attach images inside the history object
  /// for multimodal input.
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

  /// Disposes the LLM isolate and frees native resources.
  void dispose() {
    _isolate.dispose();
  }
}
