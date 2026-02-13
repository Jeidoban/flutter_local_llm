import 'dart:async';
import 'dart:convert';
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
/// final llm = await FlutterLocalLlm.init();
/// await for (final token in llm.sendMessage('Hello!')) {
///   print(token);
/// }
/// llm.dispose();
/// ```
class FlutterLocalLlm {
  final LLMIsolate _isolate;
  final LLMConfig _config;
  final int keepRecentPairs;
  int _nextRequestId = 0;

  // List of all chat histories
  List<LlmChatHistory> _chatHistories = [];

  // Index of the currently active chat
  int? _activeChatIndex;

  // Get the current LLM configuration
  LLMConfig get config => _config;

  // Get all chats
  List<LlmChatHistory> get chats => _chatHistories;

  // Get the currently active chat
  // Auto-creates a new chat if none exists
  LlmChatHistory get activeChat {
    if (_chatHistories.isEmpty || _activeChatIndex == null) {
      startNewChat();
    }
    return _chatHistories[_activeChatIndex!];
  }

  /// Get the index of the currently active chat
  int? get activeChatIndex => _activeChatIndex;

  // Private constructor
  FlutterLocalLlm._({
    required LLMIsolate isolate,
    required LLMConfig config,
    required this.keepRecentPairs,
  }) : _isolate = isolate,
       _config = config;

  /// Initialize FlutterLocalLlm with a model
  static Future<FlutterLocalLlm> init({
    LLMModel model = LLMModel.gemma3_1b_q5,
    String? systemPrompt,
    String? customModelUrl,
    String? customImageModelUrl,
    void Function(double progress)? onDownloadProgress,
    int contextSize = 8096,
    int nPredict = -1,
    int? nBatch,
    int? messagePairsToKeepWhenClearingContext,
    int nThreads = 8,
    double temperature = 0.7,
    int topK = 64,
    double topP = 0.95,
    double minP = 0.05,
    double penaltyRepeat = 1.1,
  }) async {
    // Migrate old model directory if it exists
    await _migrateOldModels();

    // Default batch size to context size if not specified
    final batchSize = nBatch ?? contextSize;

    // Calculate keepRecentPairs based on context size if not provided
    // 4096: 2 pairs, 8096: 4 pairs, 16384: 8 pairs, etc.
    final effectiveKeepRecentPairs =
        messagePairsToKeepWhenClearingContext ??
        (contextSize / 2048).round().clamp(1, 10);

    // Build system prompt with context size guidance if not provided
    final effectiveSystemPrompt =
        systemPrompt ??
        'You are a helpful assistant. Please keep your responses concise, '
            'answer directly what the user asks, avoid unnecessary elaboration. '
            'and limit them to an absolute maximum of ${(contextSize / 2).round()} '
            'tokens to fit within the conversation context.';

    // Create config
    final config = LLMConfig(
      model: model,
      customModelUrl: customModelUrl,
      customImageModelUrl: customImageModelUrl,
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

    // Check if we need to download image model for combined progress
    final hasImageModel =
        config.imageDownloadUrl != null && config.imageFileName != null;

    // Download model with weighted progress (0-50% or 0-100%)
    final modelPath = await _getModelPath(config.downloadUrl, config.fileName, (
      progress,
    ) {
      final weightedProgress = hasImageModel ? progress * 0.5 : progress;
      onDownloadProgress?.call(weightedProgress);
    });

    // Download image model if needed (50-100%)
    String? imageModelPath;
    if (hasImageModel) {
      imageModelPath = await _getModelPath(
        config.imageDownloadUrl!,
        config.imageFileName!,
        (progress) {
          final weightedProgress = 0.5 + (progress * 0.5);
          onDownloadProgress?.call(weightedProgress);
        },
      );
    }

    // Spawn isolate and initialize
    final isolate = await LLMIsolate.spawn(
      modelPath,
      config,
      imageModelPath: imageModelPath,
    );

    final instance = FlutterLocalLlm._(
      isolate: isolate,
      config: config,
      keepRecentPairs: effectiveKeepRecentPairs,
    );

    // Load chats from storage
    await instance._loadChatsFromJson();

    return instance;
  }

  /// Get the models directory, creating it if it doesn't exist
  static Future<Directory> _getModelsDirectory() async {
    final documentsDir = await getApplicationDocumentsDirectory();
    final modelsDir = Directory(
      path.join(documentsDir.path, 'flutter_local_llm', 'models'),
    );

    if (!modelsDir.existsSync()) {
      modelsDir.createSync(recursive: true);
    }

    return modelsDir;
  }

  /// Get the data directory, creating it if it doesn't exist
  static Future<Directory> _getDataDirectory() async {
    final documentsDir = await getApplicationDocumentsDirectory();
    final dataDir = Directory(
      path.join(documentsDir.path, 'flutter_local_llm', 'data'),
    );

    if (!dataDir.existsSync()) {
      dataDir.createSync(recursive: true);
    }

    return dataDir;
  }

  /// Migrate models from old directory structure to new structure
  static Future<void> _migrateOldModels() async {
    final documentsDir = await getApplicationDocumentsDirectory();
    final oldDir = Directory(
      path.join(documentsDir.path, 'flutter_local_llm_models'),
    );

    if (oldDir.existsSync()) {
      final newDir = await _getModelsDirectory();

      // Move files from old to new
      await for (final entity in oldDir.list()) {
        if (entity is File) {
          final newPath = path.join(newDir.path, path.basename(entity.path));
          await entity.copy(newPath);
        }
      }

      // Delete old directory
      await oldDir.delete(recursive: true);
    }
  }

  /// Get the chat storage file
  Future<File> _getChatStorageFile() async {
    final dataDir = await _getDataDirectory();
    return File(path.join(dataDir.path, 'chats.json'));
  }

  /// Save all chats to JSON file
  Future<void> _saveChatsToJson() async {
    final file = await _getChatStorageFile();
    final data = {
      'activeChatIndex': _activeChatIndex,
      'chats': _chatHistories.map((chat) => chat.toJson()).toList(),
    };

    final jsonString = const JsonEncoder.withIndent('  ').convert(data);
    await file.writeAsString(jsonString);
  }

  /// Load chats from JSON file
  Future<void> _loadChatsFromJson() async {
    final file = await _getChatStorageFile();

    if (!file.existsSync()) {
      return;
    }

    try {
      final jsonString = await file.readAsString();
      final data = jsonDecode(jsonString) as Map<String, dynamic>;

      _activeChatIndex = data['activeChatIndex'] as int?;
      final chatsJson = data['chats'] as List<dynamic>?;

      if (chatsJson != null) {
        _chatHistories = chatsJson
            .map(
              (json) => LlmChatHistory.fromJson(json as Map<String, dynamic>),
            )
            .toList();
      }
    } catch (e) {
      // If loading fails, just start fresh
      _chatHistories = [];
      _activeChatIndex = null;
    }
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
    activeChat.messages.clear();
    activeChat.fullHistory.clear();

    if (_config.systemPrompt != null && _config.systemPrompt!.isNotEmpty) {
      activeChat.addMessage(role: Role.system, content: _config.systemPrompt!);
    }

    _isolate.sendCommand(ClearContextCommand());

    // Save to storage
    await _saveChatsToJson();
  }

  /// Start a new chat session
  ///
  /// Creates a new chat with an optional [title] (defaults to "New Chat").
  /// The new chat becomes the active chat.
  /// Returns the index of the new chat.
  int startNewChat({String? title}) {
    final newChat = LlmChatHistory(
      title: title ?? 'New Chat',
      createdAt: DateTime.now(),
      updatedAt: DateTime.now(),
    );

    // Add system prompt if configured
    if (_config.systemPrompt != null && _config.systemPrompt!.isNotEmpty) {
      newChat.addMessage(role: Role.system, content: _config.systemPrompt!);
    }

    _chatHistories.add(newChat);
    _activeChatIndex = _chatHistories.length - 1;

    return _activeChatIndex!;
  }

  /// Set the active chat by index
  ///
  /// Clears the LLM context and switches to the specified chat.
  void setActiveChat(int index) {
    if (index < 0 || index >= _chatHistories.length) {
      throw ArgumentError('Chat index out of bounds: $index');
    }

    _activeChatIndex = index;
    _isolate.sendCommand(ClearContextCommand());
  }

  /// Delete a chat by index
  ///
  /// If the deleted chat was active, switches to the first chat (or none if empty).
  /// Saves changes immediately.
  Future<void> deleteChat(int index) async {
    if (index < 0 || index >= _chatHistories.length) {
      throw ArgumentError('Chat index out of bounds: $index');
    }

    _chatHistories.removeAt(index);

    if (_chatHistories.isEmpty) {
      _activeChatIndex = null;
    } else {
      // Adjust active index if needed
      if (_activeChatIndex == index) {
        _activeChatIndex = 0;
        _isolate.sendCommand(ClearContextCommand());
      } else if (_activeChatIndex != null && _activeChatIndex! > index) {
        _activeChatIndex = _activeChatIndex! - 1;
      }
    }

    await _saveChatsToJson();
  }

  /// Delete all chats
  ///
  /// Clears all chat histories and the LLM context.
  /// Deletes the chat storage file.
  Future<void> deleteAllChats() async {
    _chatHistories.clear();
    _activeChatIndex = null;

    _isolate.sendCommand(ClearContextCommand());

    final file = await _getChatStorageFile();
    if (file.existsSync()) {
      await file.delete();
    }
  }

  /// Save all chats to storage
  ///
  /// Call this after manually modifying chat titles or other properties.
  Future<void> saveChats() async {
    // Update updatedAt timestamp on all chats
    for (final chat in _chatHistories) {
      chat.updatedAt = DateTime.now();
    }
    await _saveChatsToJson();
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
      _config.chatFormat,
      leaveLastAssistantOpen: true,
    );

    // Check if context is empty (just cleared or first message)
    final contextIsEmpty = remainingSpace >= _config.contextSize - 100;

    // Count images in new messages for token estimation
    final newMessageImageCount = messages.messages
        .expand((msg) => msg.images)
        .length;

    // Determine if we need to rebuild context with history
    final needsHistoryRebuild =
        contextIsEmpty ||
        activeChat.shouldTrimBeforePromptNoLlama(
          newPrompt,
          remainingSpace,
          _config.contextSize,
          imageCount: newMessageImageCount,
        );

    if (needsHistoryRebuild) {
      // Create test prompt with full history to check if it fits
      final testMessages = LlmChatHistory();
      testMessages.messages.addAll(activeChat.messages);
      testMessages.messages.addAll(messages.messages);
      final fullPrompt = testMessages.exportFormat(
        _config.chatFormat,
        leaveLastAssistantOpen: true,
      );

      // Count total images (history + new messages)
      final totalImageCount = testMessages.messages
          .expand((msg) => msg.images)
          .length;

      // Trim history if full prompt doesn't fit
      if (activeChat.shouldTrimBeforePromptNoLlama(
        fullPrompt,
        remainingSpace,
        _config.contextSize,
        imageCount: totalImageCount,
      )) {
        activeChat.autoTrimForSpaceNoLlama(keepRecentPairs: keepRecentPairs);
      }

      // Clear context and add (possibly trimmed) history
      _isolate.sendCommand(ClearContextCommand());
      tempMessages.messages.addAll(activeChat.messages);
    }

    tempMessages.messages.addAll(messages.messages);

    var imagePaths = tempMessages.messages
        .expand((message) => message.images)
        .toList();

    final promptToSend = tempMessages.exportFormat(
      _config.chatFormat,
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
      if (activeChat.title == 'New Chat' && firstUserMessage != null) {
        activeChat.title = firstUserMessage.length > 40
            ? '${firstUserMessage.substring(0, 40)}...'
            : firstUserMessage;
      }

      for (final msg in messages.messages) {
        activeChat.addMessage(
          role: msg.role,
          content: msg.content,
          images: msg.images,
        );
      }

      activeChat.addMessage(
        role: Role.assistant,
        content: responseBuffer.toString().trim(),
      );

      // Update timestamp and save
      activeChat.updatedAt = DateTime.now();
      await _saveChatsToJson();
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
