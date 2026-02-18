import 'dart:convert';
import 'dart:io';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'llm_chat_history.dart';

/// Manages chat persistence
class ChatStorage {
  final String? storagePath;

  ChatStorage({this.storagePath});

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

  /// Load all chats from storage
  Future<ChatStorageData?> loadChats() async {
    final file = await _getStorageFile();

    if (!file.existsSync()) {
      return null;
    }

    try {
      final jsonString = await file.readAsString();
      final data = jsonDecode(jsonString) as Map<String, dynamic>;

      final activeChatIndex = data['activeChatIndex'] as int?;
      final chatsJson = data['chats'] as List<dynamic>?;

      List<LlmChatHistory> chats = [];
      if (chatsJson != null) {
        chats = chatsJson
            .map(
              (json) => LlmChatHistory.fromJson(json as Map<String, dynamic>),
            )
            .toList();
      }

      return ChatStorageData(activeChatIndex: activeChatIndex, chats: chats);
    } catch (e) {
      return null;
    }
  }

  /// Save all chats to storage
  Future<void> saveChats(ChatStorageData data) async {
    final file = await _getStorageFile();
    final jsonData = {
      'activeChatIndex': data.activeChatIndex,
      'chats': data.chats.map((chat) => chat.toJson()).toList(),
    };

    final jsonString = const JsonEncoder.withIndent('  ').convert(jsonData);
    await file.writeAsString(jsonString);
  }

  /// Delete chat storage
  Future<void> deleteStorage() async {
    final file = await _getStorageFile();
    if (file.existsSync()) {
      await file.delete();
    }
  }
}

/// Data structure for chat persistence
class ChatStorageData {
  final int? activeChatIndex;
  final List<LlmChatHistory> chats;

  ChatStorageData({this.activeChatIndex, required this.chats});
}
