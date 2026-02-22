import 'dart:convert';
import 'dart:io';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'llm_chat_history.dart';

/// Manages chat state and persistence
class ChatManager {
  String? storagePath;
  void Function()? onSessionChanged;
  int keepRecentPairs;

  // Chat state
  List<LlmChatHistory> _chats = [];
  int? _activeChatIndex;

  ChatManager({
    this.storagePath,
    this.onSessionChanged,
    this.keepRecentPairs = 2,
  });

  /// Get all chats
  List<LlmChatHistory> get chats => _chats;

  /// Get the index of the currently active chat
  int? get activeChatIndex => _activeChatIndex;

  set activeChatIndex(int index) {
    if (index < 0 || index >= _chats.length) {
      throw ArgumentError('Chat index out of bounds: $index');
    }
    _activeChatIndex = index;
    onSessionChanged?.call();
  }

  /// Get the currently active chat (auto-creates if none exists)
  LlmChatHistory get activeChat {
    if (_chats.isEmpty) startNewChat();
    _activeChatIndex ??= 0;
    return _chats[_activeChatIndex!];
  }

  /// Load chats from storage
  Future<void> loadChats() async {
    final file = await _getStorageFile();

    if (!file.existsSync()) {
      return;
    }

    try {
      final jsonString = await file.readAsString();
      final data = jsonDecode(jsonString) as Map<String, dynamic>;

      _activeChatIndex = data['activeChatIndex'] as int?;
      final chatsJson = data['chats'] as List<dynamic>?;

      if (chatsJson != null) {
        _chats = chatsJson
            .map(
              (json) => LlmChatHistory.fromJson(json as Map<String, dynamic>),
            )
            .toList();
      }
    } catch (e) {
      // Ignore errors and start fresh
    }
  }

  /// Start a new chat session
  ///
  /// Creates a new chat with an optional [title] (defaults to "New Chat") and
  /// [systemPrompt] (defaults to "You are a helpful assistant.").
  /// The new chat becomes the active chat.
  /// Returns the index of the new chat.
  int startNewChat({
    String title = 'New Chat',
    String systemPrompt = 'You are a helpful assistant.',
  }) {
    final newChat = LlmChatHistory(
      title: title,
      createdAt: DateTime.now(),
      updatedAt: DateTime.now(),
    );

    if (systemPrompt.isNotEmpty) {
      newChat.addMessage(role: Role.system, content: systemPrompt);
    }

    _chats.add(newChat);
    _activeChatIndex = _chats.length - 1;

    return _activeChatIndex!;
  }

  /// Delete a chat by index
  ///
  /// If the deleted chat was active, switches to the first chat (or none if empty).
  /// Saves changes immediately.
  Future<void> deleteChat(int index) async {
    if (index < 0 || index >= _chats.length) {
      throw ArgumentError('Chat index out of bounds: $index');
    }

    _chats.removeAt(index);

    if (_chats.isEmpty) {
      _activeChatIndex = null;
    } else {
      // Adjust active index if needed
      if (_activeChatIndex == index) {
        _activeChatIndex = 0;
        onSessionChanged?.call();
      } else if (_activeChatIndex != null && _activeChatIndex! > index) {
        _activeChatIndex = _activeChatIndex! - 1;
      }
    }

    await saveChats();
  }

  /// Delete all chats
  ///
  /// Clears all chat histories and deletes the storage file.
  Future<void> deleteAllChats() async {
    _chats.clear();
    _activeChatIndex = null;
    onSessionChanged?.call();

    final file = await _getStorageFile();
    if (file.existsSync()) {
      await file.delete();
    }
  }

  /// Save all chats to storage
  ///
  /// Updates all chat timestamps before saving.
  Future<void> saveChats() async {
    // Update updatedAt timestamp on all chats
    for (final chat in _chats) {
      chat.updatedAt = DateTime.now();
    }

    final file = await _getStorageFile();
    final jsonData = {
      'activeChatIndex': _activeChatIndex,
      'chats': _chats.map((chat) => chat.toJson()).toList(),
    };

    final jsonString = const JsonEncoder.withIndent('  ').convert(jsonData);
    await file.writeAsString(jsonString);
  }

  Future<File> _getStorageFile() async {
    if (storagePath != null) {
      return File(storagePath!);
    }

    // Default: app documents directory
    final documentsDir = await getApplicationDocumentsDirectory();
    final dataDir = Directory(
      path.join(documentsDir.path, 'flutter_local_llm', 'data'),
    );
    if (!dataDir.existsSync()) {
      dataDir.createSync(recursive: true);
    }
    return File(path.join(dataDir.path, 'chats.json'));
  }
}
