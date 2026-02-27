import 'dart:convert';
import 'dart:io';
import 'package:flutter_local_llm/src/managers/chat_manager.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import '../test_helpers.dart';

void main() {
  group('ChatManager startNewChat()', () {
    test('creates a chat with default title and system prompt', () {
      final manager = ChatManager();
      manager.startNewChat();

      expect(manager.chats.length, 1);
      expect(manager.chats[0].title, 'New Chat');
      expect(manager.chats[0].messages.length, 1);
      expect(manager.chats[0].messages[0].role, Role.system);
      expect(
        manager.chats[0].messages[0].content,
        'You are a helpful assistant.',
      );
    });

    test('creates a chat with a custom title and system prompt', () {
      final manager = ChatManager();
      manager.startNewChat(title: 'My Chat', systemPrompt: 'Be brief.');

      expect(manager.chats[0].title, 'My Chat');
      expect(manager.chats[0].messages[0].content, 'Be brief.');
    });

    test('skips system message when systemPrompt is empty', () {
      final manager = ChatManager();
      manager.startNewChat(systemPrompt: '');

      expect(manager.chats[0].messages.isEmpty, true);
    });

    test('makes the new chat active and returns its index', () {
      final manager = ChatManager();
      manager.startNewChat();
      final index = manager.startNewChat();

      expect(manager.chats.length, 2);
      expect(index, 1);
      expect(manager.activeChatIndex, 1);
    });
  });

  group('ChatManager activeChat', () {
    test('auto-creates a chat when the list is empty', () {
      final manager = ChatManager();
      expect(manager.chats.isEmpty, true);

      final chat = manager.activeChat;

      expect(manager.chats.length, 1);
      expect(chat, isNotNull);
    });

    test('returns the chat at activeChatIndex', () {
      final manager = ChatManager();
      manager.startNewChat(title: 'First');
      manager.startNewChat(title: 'Second');
      manager.activeChatIndex = 0;

      expect(manager.activeChat.title, 'First');
    });
  });

  group('ChatManager activeChatIndex setter', () {
    test('switches active chat and fires onSessionChanged', () {
      var callCount = 0;
      final manager = ChatManager(onSessionChanged: () => callCount++);
      manager.startNewChat();
      manager.startNewChat();

      manager.activeChatIndex = 0;

      expect(manager.activeChatIndex, 0);
      expect(callCount, 1);
    });

    test('throws ArgumentError for out-of-bounds index', () {
      final manager = ChatManager();
      manager.startNewChat();

      expect(() => manager.activeChatIndex = 5, throwsArgumentError);
      expect(() => manager.activeChatIndex = -1, throwsArgumentError);
    });
  });

  group('ChatManager deleteChat()', () {
    test('removes the chat at the given index', () {
      final manager = ChatManager(storagePath: '/dev/null');
      manager.startNewChat(title: 'A');
      manager.startNewChat(title: 'B');
      manager.startNewChat(title: 'C');

      manager.deleteChat(1);

      expect(manager.chats.length, 2);
      expect(manager.chats[0].title, 'A');
      expect(manager.chats[1].title, 'C');
    });

    test('switches active to index 0 when the active chat is deleted', () {
      var sessionChanges = 0;
      final manager = ChatManager(
        storagePath: '/dev/null',
        onSessionChanged: () => sessionChanges++,
      );
      manager.startNewChat(title: 'A');
      manager.startNewChat(title: 'B');
      manager.activeChatIndex = 1;
      sessionChanges = 0;

      manager.deleteChat(1);

      expect(manager.activeChatIndex, 0);
      expect(sessionChanges, 1);
    });

    test('adjusts active index when a preceding chat is deleted', () {
      final manager = ChatManager(storagePath: '/dev/null');
      manager.startNewChat(title: 'A');
      manager.startNewChat(title: 'B');
      manager.startNewChat(title: 'C');
      manager.activeChatIndex = 2;

      manager.deleteChat(0);

      expect(manager.activeChatIndex, 1);
      expect(manager.activeChat.title, 'C');
    });

    test('does not adjust active index when a following chat is deleted', () {
      final manager = ChatManager(storagePath: '/dev/null');
      manager.startNewChat(title: 'A');
      manager.startNewChat(title: 'B');
      manager.startNewChat(title: 'C');
      manager.activeChatIndex = 0;

      manager.deleteChat(2);

      expect(manager.activeChatIndex, 0);
    });

    test('sets activeChatIndex to null when the last chat is deleted', () {
      final manager = ChatManager(storagePath: '/dev/null');
      manager.startNewChat();

      manager.deleteChat(0);

      expect(manager.chats.isEmpty, true);
      expect(manager.activeChatIndex, isNull);
    });

    test('throws ArgumentError for out-of-bounds index', () {
      final manager = ChatManager();
      manager.startNewChat();

      expect(() => manager.deleteChat(5), throwsArgumentError);
      expect(() => manager.deleteChat(-1), throwsArgumentError);
    });
  });

  group('ChatManager deleteAllChats()', () {
    test('clears all chats and fires onSessionChanged', () async {
      var sessionChanges = 0;
      final tempDir = await Directory.systemTemp.createTemp('cm_clear_test_');
      addTearDown(() => tempDir.delete(recursive: true));
      final manager = ChatManager(
        storagePath: '${tempDir.path}/chats.json',
        onSessionChanged: () => sessionChanges++,
      );
      manager.startNewChat();
      manager.startNewChat();

      await manager.deleteAllChats();

      expect(manager.chats.isEmpty, true);
      expect(manager.activeChatIndex, isNull);
      expect(sessionChanges, 1);
    });

    test('deletes the storage file', () async {
      final tempDir = await Directory.systemTemp.createTemp('cm_test_');
      final storagePath = '${tempDir.path}/chats.json';
      addTearDown(() => tempDir.delete(recursive: true));

      final manager = ChatManager(storagePath: storagePath);
      manager.startNewChat();
      await manager.saveChats();
      expect(File(storagePath).existsSync(), true);

      await manager.deleteAllChats();

      expect(File(storagePath).existsSync(), false);
    });
  });

  group('ChatManager saveChats() / loadChats()', () {
    late TempTestDirectory tempDir;

    setUp(() async {
      tempDir = TempTestDirectory();
      await tempDir.setup();
    });

    tearDown(() async {
      await tempDir.teardown();
    });

    test('round-trips chats with full history and active index', () async {
      final storagePath = '${tempDir.dir.path}/chats.json';
      final manager = ChatManager(storagePath: storagePath);

      manager.startNewChat(title: 'Chat One', systemPrompt: 'Prompt A');
      manager.startNewChat(title: 'Chat Two', systemPrompt: 'Prompt B');
      manager.chats[0].addMessage(role: Role.user, content: 'Hello');
      manager.chats[0].addMessage(role: Role.assistant, content: 'Hi');
      manager.activeChatIndex = 1;

      await manager.saveChats();

      final loaded = ChatManager(storagePath: storagePath);
      await loaded.loadChats();

      expect(loaded.chats.length, 2);
      expect(loaded.chats[0].title, 'Chat One');
      expect(loaded.chats[1].title, 'Chat Two');
      expect(loaded.activeChatIndex, 1);
      expect(loaded.chats[0].fullHistory.length, 3);
    });

    test('does nothing when storage file does not exist', () async {
      final storagePath = '${tempDir.dir.path}/nonexistent.json';
      final manager = ChatManager(storagePath: storagePath);

      await manager.loadChats();

      expect(manager.chats.isEmpty, true);
      expect(manager.activeChatIndex, isNull);
    });

    test('silently ignores corrupt JSON', () async {
      final storagePath = '${tempDir.dir.path}/corrupt.json';
      File(storagePath).writeAsStringSync('not valid json {{{');

      final manager = ChatManager(storagePath: storagePath);
      await manager.loadChats();

      expect(manager.chats.isEmpty, true);
    });

    test('saveChats updates timestamps on all chats', () async {
      final storagePath = '${tempDir.dir.path}/chats.json';
      final manager = ChatManager(storagePath: storagePath);
      manager.startNewChat();

      final before = DateTime.now();
      await manager.saveChats();
      final after = DateTime.now();

      final json = jsonDecode(File(storagePath).readAsStringSync())
          as Map<String, dynamic>;
      final updatedAt = DateTime.parse(
        (json['chats'] as List).first['updatedAt'] as String,
      );

      expect(updatedAt.isAfter(before) || timestampsNear(updatedAt, before), true);
      expect(updatedAt.isBefore(after) || timestampsNear(updatedAt, after), true);
    });

    test('saves activeChatIndex as null when no chats exist', () async {
      final storagePath = '${tempDir.dir.path}/chats.json';
      final manager = ChatManager(storagePath: storagePath);

      await manager.saveChats();

      final json = jsonDecode(File(storagePath).readAsStringSync())
          as Map<String, dynamic>;
      expect(json['activeChatIndex'], isNull);
      expect(json['chats'], isEmpty);
    });
  });

  group('ChatManager storagePath', () {
    test('uses custom storagePath when provided', () async {
      final tempDir = await Directory.systemTemp.createTemp('cm_path_test_');
      final storagePath = '${tempDir.path}/custom_chats.json';
      addTearDown(() => tempDir.delete(recursive: true));

      final manager = ChatManager(storagePath: storagePath);
      manager.startNewChat();
      await manager.saveChats();

      expect(File(storagePath).existsSync(), true);
    });
  });
}
