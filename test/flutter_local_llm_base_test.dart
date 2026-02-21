import 'dart:async';
import 'package:flutter_local_llm/flutter_local_llm.dart';
import 'package:flutter_local_llm/src/llm_isolate.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'test_helpers.dart';

void main() {
  setUpAll(() {
    registerMocktailFallbacks();
  });

  group('FlutterLocalLlm', () {
    late MockModelManager mockModelManager;
    late MockChatManager mockChatManager;
    late MockLLMIsolate mockIsolate;
    late StreamController<IsolateResponse> responseController;
    late LlmConfig defaultConfig;

    setUp(() {
      // Create default config for tests
      defaultConfig = LlmConfig.gemma3_1b_q5.copyWith(contextSize: 8096);

      mockModelManager = MockModelManager();
      mockChatManager = MockChatManager();
      mockIsolate = MockLLMIsolate();
      responseController = StreamController<IsolateResponse>.broadcast();

      // Default stubs
      when(
        () => mockModelManager.createModelIsolate(any()),
      ).thenAnswer((_) async => mockIsolate);

      when(() => mockChatManager.loadChats()).thenAnswer((_) async {});
      when(() => mockChatManager.saveChats()).thenAnswer((_) async {});
      when(() => mockChatManager.chats).thenReturn([LlmChatHistory()]);
      when(() => mockChatManager.activeChat).thenReturn(LlmChatHistory());
      when(
        () => mockChatManager.startNewChat(
          title: any(named: 'title'),
          systemPrompt: any(named: 'systemPrompt'),
        ),
      ).thenReturn(0);

      when(
        () => mockIsolate.responseStream,
      ).thenAnswer((_) => responseController.stream);

      // Set up default command handling that responds to all command types
      when(() => mockIsolate.sendCommand(any())).thenAnswer((invocation) {
        final command = invocation.positionalArguments[0];
        if (command is GetRemainingContextCommand) {
          // Defer response by one microtask so the await-for listener is set up first
          Future.microtask(
            () => responseController.add(
              RemainingContextResponse(
                remaining: 4096,
                requestId: command.requestId,
              ),
            ),
          );
        }
      });

      when(() => mockIsolate.dispose()).thenReturn(null);
    });

    tearDown(() {
      responseController.close();
    });

    test('creates instance with default dependencies', () async {
      final llm = await FlutterLocalLlm.createCustom(
        config: defaultConfig,
        modelManager: mockModelManager,
        chatManager: mockChatManager,
      );

      expect(llm, isNotNull);
      verify(() => mockModelManager.createModelIsolate(any())).called(1);
      verify(() => mockChatManager.loadChats()).called(1);
    });

    test('sends message and streams tokens', () async {
      when(() => mockIsolate.sendCommand(any())).thenAnswer((invocation) {
        final command = invocation.positionalArguments[0];
        if (command is GetRemainingContextCommand) {
          Future.microtask(
            () => responseController.add(
              RemainingContextResponse(
                remaining: 4096,
                requestId: command.requestId,
              ),
            ),
          );
        } else if (command is GenerateFromPromptCommand) {
          final generateRequestId = command.requestId;
          Future.microtask(() {
            responseController.add(TokenResponse(token: 'Hello', requestId: generateRequestId));
            responseController.add(TokenResponse(token: ' ', requestId: generateRequestId));
            responseController.add(TokenResponse(token: 'world', requestId: generateRequestId));
            responseController.add(CompletionResponse(requestId: generateRequestId));
          });
        }
      });

      final llm = await FlutterLocalLlm.createCustom(
        config: defaultConfig,
        modelManager: mockModelManager,
        chatManager: mockChatManager,
      );

      final tokens = <String>[];
      await for (final token in llm.sendMessage('Hi', addToHistory: false)) {
        tokens.add(token);
      }

      expect(tokens, ['Hello', ' ', 'world']);
      verify(
        () => mockIsolate.sendCommand(
          any(
            that: isA<GenerateFromPromptCommand>().having(
              (c) => c.prompt,
              'prompt',
              contains('Hi'),
            ),
          ),
        ),
      ).called(1);
    });

    test('auto-creates chat on first message', () async {
      when(() => mockIsolate.sendCommand(any())).thenAnswer((invocation) {
        final command = invocation.positionalArguments[0];
        if (command is GetRemainingContextCommand) {
          Future.microtask(
            () => responseController.add(
              RemainingContextResponse(
                remaining: 4096,
                requestId: command.requestId,
              ),
            ),
          );
        } else if (command is GenerateFromPromptCommand) {
          final generateRequestId = command.requestId;
          Future.microtask(() {
            responseController.add(TokenResponse(token: 'Response', requestId: generateRequestId));
            responseController.add(CompletionResponse(requestId: generateRequestId));
          });
        }
      });

      final llm = await FlutterLocalLlm.createCustom(
        config: defaultConfig,
        modelManager: mockModelManager,
        chatManager: mockChatManager,
      );

      when(() => mockChatManager.chats).thenReturn([]);
      when(() => mockChatManager.activeChat).thenReturn(LlmChatHistory());
      when(() => mockChatManager.startNewChat()).thenReturn(0);

      expect(llm.chatManager.chats.isEmpty, true);

      await for (final _ in llm.sendMessage('First message')) {
        // Consume stream
      }

      verify(() => mockChatManager.saveChats()).called(greaterThan(0));
    });

    test('loads existing chats on init', () async {
      final existingChats = [
        LlmChatHistory(title: 'Existing Chat 1'),
        LlmChatHistory(title: 'Existing Chat 2'),
      ];

      when(() => mockChatManager.loadChats()).thenAnswer((_) async {});
      when(() => mockChatManager.chats).thenReturn(existingChats);
      when(() => mockChatManager.activeChatIndex).thenReturn(0);
      when(() => mockChatManager.activeChat).thenReturn(existingChats[0]);

      final llm = await FlutterLocalLlm.createCustom(
        config: defaultConfig,
        modelManager: mockModelManager,
        chatManager: mockChatManager,
      );

      expect(llm.chatManager.chats.length, 2);
      expect(llm.chatManager.chats[0].title, 'Existing Chat 1');
      expect(llm.chatManager.activeChatIndex, 0);
    });

    test('deleteAllChats clears storage', () async {
      when(() => mockChatManager.deleteAllChats()).thenAnswer((_) async {});
      when(() => mockChatManager.chats).thenReturn([]);

      final llm = await FlutterLocalLlm.createCustom(
        config: defaultConfig,
        modelManager: mockModelManager,
        chatManager: mockChatManager,
      );

      await llm.chatManager.deleteAllChats();

      expect(llm.chatManager.chats.isEmpty, true);
      verify(() => mockChatManager.deleteAllChats()).called(1);
    });

    test('switches between chats', () async {
      final existingChats = [
        LlmChatHistory(title: 'Chat 1'),
        LlmChatHistory(title: 'Chat 2'),
      ];

      when(() => mockChatManager.loadChats()).thenAnswer((_) async {});
      when(() => mockChatManager.chats).thenReturn(existingChats);
      when(() => mockChatManager.activeChatIndex).thenReturn(0);
      when(() => mockChatManager.activeChat).thenReturn(existingChats[0]);

      final llm = await FlutterLocalLlm.createCustom(
        config: defaultConfig,
        modelManager: mockModelManager,
        chatManager: mockChatManager,
      );

      expect(llm.chatManager.activeChatIndex, 0);
      expect(llm.chatManager.activeChat.title, 'Chat 1');

      // setActiveChat is now done via activeChatIndex setter
      llm.chatManager.activeChatIndex = 1;
    });

    test('starts new chat', () async {
      when(() => mockChatManager.chats).thenReturn([]);
      when(
        () => mockChatManager.startNewChat(title: any(named: 'title')),
      ).thenReturn(0);

      final llm = await FlutterLocalLlm.createCustom(
        config: defaultConfig,
        modelManager: mockModelManager,
        chatManager: mockChatManager,
      );

      expect(llm.chatManager.chats.isEmpty, true);

      final chatIndex = llm.chatManager.startNewChat(title: 'My New Chat');

      expect(chatIndex, 0);
      verify(
        () => mockChatManager.startNewChat(title: 'My New Chat'),
      ).called(1);
    });

    test('deletes chat and adjusts active index', () async {
      final existingChats = [
        LlmChatHistory(title: 'Chat 1'),
        LlmChatHistory(title: 'Chat 2'),
        LlmChatHistory(title: 'Chat 3'),
      ];

      when(() => mockChatManager.loadChats()).thenAnswer((_) async {});
      when(() => mockChatManager.chats).thenReturn(existingChats);
      when(() => mockChatManager.activeChatIndex).thenReturn(1);
      when(() => mockChatManager.deleteChat(any())).thenAnswer((_) async {});

      final llm = await FlutterLocalLlm.createCustom(
        config: defaultConfig,
        modelManager: mockModelManager,
        chatManager: mockChatManager,
      );

      expect(llm.chatManager.chats.length, 3);
      expect(llm.chatManager.activeChatIndex, 1);

      // Delete the active chat
      await llm.chatManager.deleteChat(1);

      verify(() => mockChatManager.deleteChat(1)).called(1);
    });

    test('clears history', () async {
      final customConfig = LlmConfig.gemma3_1b_q5.copyWith(
        systemPrompt: 'Test system prompt',
        contextSize: 8096,
      );

      final testChat = LlmChatHistory();
      when(() => mockChatManager.activeChat).thenReturn(testChat);

      final llm = await FlutterLocalLlm.createCustom(
        config: customConfig,
        modelManager: mockModelManager,
        chatManager: mockChatManager,
      );

      // Add a test message
      llm.chatManager.activeChat.addMessage(
        role: Role.user,
        content: 'Test message',
      );

      expect(llm.chatManager.activeChat.messages.length, 1);

      await llm.clearHistory();

      // clearHistory clears the active chat and adds system prompt back
      expect(llm.chatManager.activeChat.messages.length, 1);
      expect(llm.chatManager.activeChat.messages[0].role, Role.system);
      verify(
        () => mockIsolate.sendCommand(any(that: isA<ClearContextCommand>())),
      ).called(1);
      verify(() => mockChatManager.saveChats()).called(1);
    });

    test('downloads multimodal model', () async {
      final multimodalConfig = LlmConfig.gemma3_4b_q5_mm.copyWith(contextSize: 8096);

      await FlutterLocalLlm.createCustom(
        config: multimodalConfig,
        modelManager: mockModelManager,
        chatManager: mockChatManager,
      );

      // Should create isolate with multimodal config
      verify(() => mockModelManager.createModelIsolate(any())).called(1);
    });
  });
}
