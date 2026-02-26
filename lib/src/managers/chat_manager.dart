import 'dart:convert';
import 'dart:io';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import '../models/llm_chat_history.dart';

/// Manages chat state and persistence.
///
/// Owns the list of [LlmChatHistory] objects, tracks which chat is active,
/// and serializes everything to a single JSON file on disk.
///
/// Pass [storagePath] to override the default file location
/// (`<app_documents>/flutter_local_llm/data/chats.json`).
/// Set [onSessionChanged] to be notified when the active chat switches —
/// typically used to clear the LLM context to prevent history from bleeding
/// between sessions.
class ChatManager {
  String? storagePath;
  void Function()? onSessionChanged;
  int keepRecentPairs;

  List<LlmChatHistory> _chats = [];
  int? _activeChatIndex;

  ChatManager({
    this.storagePath,
    this.onSessionChanged,
    this.keepRecentPairs = 2,
  });

  /// All loaded chat histories.
  List<LlmChatHistory> get chats => _chats;

  /// The index into [chats] of the currently active chat, or null if no chats exist.
  int? get activeChatIndex => _activeChatIndex;

  /// Switches the active chat to [index] and fires [onSessionChanged].
  ///
  /// Throws [ArgumentError] if [index] is out of bounds.
  set activeChatIndex(int index) {
    if (index < 0 || index >= _chats.length) {
      throw ArgumentError('Chat index out of bounds: $index');
    }
    _activeChatIndex = index;
    onSessionChanged?.call();
  }

  /// The currently active [LlmChatHistory].
  ///
  /// Auto-creates a new chat with default settings if none exists.
  LlmChatHistory get activeChat {
    if (_chats.isEmpty) startNewChat();
    _activeChatIndex ??= 0;
    return _chats[_activeChatIndex!];
  }

  /// Loads chats from the storage file, replacing any in-memory state.
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

  /// Creates a new chat and makes it active, returning its index.
  ///
  /// [title] defaults to `"New Chat"`. [systemPrompt] is added as the first
  /// message if non-empty.
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

  /// Deletes the chat at [index] and saves the updated list.
  ///
  /// If the deleted chat was active, switches to index 0. Adjusts
  /// [activeChatIndex] to keep it pointing at the same logical chat when
  /// a chat before it is removed. Throws [ArgumentError] if out of bounds.
  Future<void> deleteChat(int index) async {
    if (index < 0 || index >= _chats.length) {
      throw ArgumentError('Chat index out of bounds: $index');
    }

    _chats.removeAt(index);

    if (_chats.isEmpty) {
      _activeChatIndex = null;
    } else {
      if (_activeChatIndex == index) {
        _activeChatIndex = 0;
        onSessionChanged?.call();
      } else if (_activeChatIndex != null && _activeChatIndex! > index) {
        _activeChatIndex = _activeChatIndex! - 1;
      }
    }

    await saveChats();
  }

  /// Clears all chats from memory and deletes the storage file.
  Future<void> deleteAllChats() async {
    _chats.clear();
    _activeChatIndex = null;
    onSessionChanged?.call();

    final file = await _getStorageFile();
    if (file.existsSync()) {
      await file.delete();
    }
  }

  /// Persists all chats to the storage file as pretty-printed JSON.
  Future<void> saveChats() async {
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

  /// Returns the [File] used for persistence, creating parent directories as needed.
  ///
  /// Uses [storagePath] if set, otherwise defaults to
  /// `<app_documents>/flutter_local_llm/data/chats.json`.
  Future<File> _getStorageFile() async {
    if (storagePath != null) {
      return File(storagePath!);
    }

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
