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
    late MockChatStorage mockChatStorage;
    late MockIsolateManager mockIsolateManager;
    late MockLLMIsolate mockIsolate;
    late StreamController<IsolateResponse> responseController;
    late LLMConfig defaultConfig;

    setUp(() {
      // Create default config for tests
      defaultConfig = LLMConfig(
        model: LLMModel.gemma3_1b_q5,
        contextSize: 8096,
        nPredict: -1,
        nBatch: 8096,
        nThreads: 8,
        temperature: 0.7,
        topK: 64,
        topP: 0.95,
        minP: 0.05,
        penaltyRepeat: 1.1,
      );

      mockModelManager = MockModelManager();
      mockChatStorage = MockChatStorage();
      mockIsolateManager = MockIsolateManager();
      mockIsolate = MockLLMIsolate();
      responseController = StreamController<IsolateResponse>.broadcast();

      // Default stubs
      when(
        () => mockModelManager.getModelPath(
          any(),
          downloadUrl: any(named: 'downloadUrl'),
        ),
      ).thenAnswer((_) async => '/fake/model.gguf');

      when(
        () => mockIsolateManager.createIsolate(
          any(),
          any(),
          imageModelPath: any(named: 'imageModelPath'),
        ),
      ).thenAnswer((_) async => mockIsolate);

      when(() => mockChatStorage.loadChats()).thenAnswer((_) async => null);

      when(() => mockChatStorage.saveChats(any())).thenAnswer((_) async {});

      when(
        () => mockIsolate.responseStream,
      ).thenAnswer((_) => responseController.stream);

      // Set up default command handling that responds to all command types
      when(() => mockIsolate.sendCommand(any())).thenAnswer((invocation) {
        final command = invocation.positionalArguments[0];
        if (command is GetRemainingContextCommand) {
          // Simulate remaining context response (add immediately since it's a broadcast stream)
          responseController.add(
            RemainingContextResponse(
              remaining: 4096,
              requestId: command.requestId,
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
      final llm = await FlutterLocalLlm.createWithDependencies(
        config: defaultConfig,
        modelManager: mockModelManager,
        chatStorage: mockChatStorage,
        isolateManager: mockIsolateManager,
      );

      expect(llm, isNotNull);
      verify(() => mockModelManager.getModelPath(any(), downloadUrl: any(named: 'downloadUrl'))).called(1);
      verify(
        () => mockIsolateManager.createIsolate(
          any(),
          any(),
          imageModelPath: any(named: 'imageModelPath'),
        ),
      ).called(1);
      verify(() => mockChatStorage.loadChats()).called(1);
    });

    test('sends message and streams tokens', () async {
      // Setup response stream
      final requestId = 0;

      when(() => mockIsolate.sendCommand(any())).thenAnswer((invocation) {
        final command = invocation.positionalArguments[0];
        if (command is GetRemainingContextCommand) {
          responseController.add(
            RemainingContextResponse(
              remaining: 4096,
              requestId: command.requestId,
            ),
          );
        } else if (command is GenerateFromPromptCommand) {
          // Simulate streaming response
          responseController.add(
            TokenResponse(token: 'Hello', requestId: requestId),
          );
          responseController.add(
            TokenResponse(token: ' ', requestId: requestId),
          );
          responseController.add(
            TokenResponse(token: 'world', requestId: requestId),
          );
          responseController.add(CompletionResponse(requestId: requestId));
        }
      });

      final llm = await FlutterLocalLlm.createWithDependencies(
        config: defaultConfig,
        modelManager: mockModelManager,
        chatStorage: mockChatStorage,
        isolateManager: mockIsolateManager,
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
          responseController.add(
            RemainingContextResponse(
              remaining: 4096,
              requestId: command.requestId,
            ),
          );
        } else if (command is GenerateFromPromptCommand) {
          responseController.add(
            TokenResponse(token: 'Response', requestId: 0),
          );
          responseController.add(CompletionResponse(requestId: 0));
        }
      });

      final llm = await FlutterLocalLlm.createWithDependencies(
        config: defaultConfig,
        modelManager: mockModelManager,
        chatStorage: mockChatStorage,
        isolateManager: mockIsolateManager,
      );

      expect(llm.chats.isEmpty, true);

      await for (final _ in llm.sendMessage('First message')) {
        // Consume stream
      }

      expect(llm.chats.length, 1);
      expect(llm.chats[0].title, contains('First message'));
      verify(() => mockChatStorage.saveChats(any())).called(greaterThan(0));
    });

    test('loads existing chats on init', () async {
      final existingChats = [
        LlmChatHistory(title: 'Existing Chat 1'),
        LlmChatHistory(title: 'Existing Chat 2'),
      ];

      when(() => mockChatStorage.loadChats()).thenAnswer(
        (_) async => (activeChatIndex: 0, chats: existingChats),
      );

      final llm = await FlutterLocalLlm.createWithDependencies(
        config: defaultConfig,
        modelManager: mockModelManager,
        chatStorage: mockChatStorage,
        isolateManager: mockIsolateManager,
      );

      expect(llm.chats.length, 2);
      expect(llm.chats[0].title, 'Existing Chat 1');
      expect(llm.activeChatIndex, 0);
    });

    test('deleteAllChats clears storage', () async {
      when(() => mockChatStorage.deleteStorage()).thenAnswer((_) async {});

      final llm = await FlutterLocalLlm.createWithDependencies(
        config: defaultConfig,
        modelManager: mockModelManager,
        chatStorage: mockChatStorage,
        isolateManager: mockIsolateManager,
      );

      await llm.deleteAllChats();

      expect(llm.chats.isEmpty, true);
      verify(() => mockChatStorage.deleteStorage()).called(1);
      verify(
        () => mockIsolate.sendCommand(any(that: isA<ClearContextCommand>())),
      ).called(1);
    });

    test('switches between chats', () async {
      final existingChats = [
        LlmChatHistory(title: 'Chat 1'),
        LlmChatHistory(title: 'Chat 2'),
      ];

      when(() => mockChatStorage.loadChats()).thenAnswer(
        (_) async => (activeChatIndex: 0, chats: existingChats),
      );

      final llm = await FlutterLocalLlm.createWithDependencies(
        config: defaultConfig,
        modelManager: mockModelManager,
        chatStorage: mockChatStorage,
        isolateManager: mockIsolateManager,
      );

      expect(llm.activeChatIndex, 0);
      expect(llm.activeChat.title, 'Chat 1');

      llm.setActiveChat(1);

      expect(llm.activeChatIndex, 1);
      expect(llm.activeChat.title, 'Chat 2');
      verify(
        () => mockIsolate.sendCommand(any(that: isA<ClearContextCommand>())),
      ).called(1);
    });

    test('starts new chat', () async {
      final llm = await FlutterLocalLlm.createWithDependencies(
        config: defaultConfig,
        modelManager: mockModelManager,
        chatStorage: mockChatStorage,
        isolateManager: mockIsolateManager,
      );

      expect(llm.chats.isEmpty, true);

      final chatIndex = llm.startNewChat(title: 'My New Chat');

      expect(llm.chats.length, 1);
      expect(llm.chats[0].title, 'My New Chat');
      expect(chatIndex, 0);
      expect(llm.activeChatIndex, 0);
    });

    test('deletes chat and adjusts active index', () async {
      final existingChats = [
        LlmChatHistory(title: 'Chat 1'),
        LlmChatHistory(title: 'Chat 2'),
        LlmChatHistory(title: 'Chat 3'),
      ];

      when(() => mockChatStorage.loadChats()).thenAnswer(
        (_) async => (activeChatIndex: 1, chats: existingChats),
      );

      final llm = await FlutterLocalLlm.createWithDependencies(
        config: defaultConfig,
        modelManager: mockModelManager,
        chatStorage: mockChatStorage,
        isolateManager: mockIsolateManager,
      );

      expect(llm.chats.length, 3);
      expect(llm.activeChatIndex, 1);

      // Delete the active chat
      await llm.deleteChat(1);

      expect(llm.chats.length, 2);
      expect(llm.activeChatIndex, 0); // Should reset to 0
      verify(() => mockChatStorage.saveChats(any())).called(greaterThan(0));
    });

    test('clears history', () async {
      final customConfig = LLMConfig(
        model: LLMModel.gemma3_1b_q5,
        systemPrompt: 'Test system prompt',
        contextSize: 8096,
        nPredict: -1,
        nBatch: 8096,
        nThreads: 8,
        temperature: 0.7,
        topK: 64,
        topP: 0.95,
        minP: 0.05,
        penaltyRepeat: 1.1,
      );

      final llm = await FlutterLocalLlm.createWithDependencies(
        config: customConfig,
        modelManager: mockModelManager,
        chatStorage: mockChatStorage,
        isolateManager: mockIsolateManager,
      );

      // Create a chat with some messages
      llm.startNewChat();
      llm.activeChat.addMessage(role: Role.user, content: 'Test message');

      expect(llm.activeChat.messages.length, 2); // System + user message

      await llm.clearHistory();

      // Should only have system message left
      expect(llm.activeChat.messages.length, 1);
      expect(llm.activeChat.messages[0].role, Role.system);
      verify(
        () => mockIsolate.sendCommand(any(that: isA<ClearContextCommand>())),
      ).called(1);
      verify(() => mockChatStorage.saveChats(any())).called(greaterThan(0));
    });

    test('downloads multimodal model', () async {
      final multimodalConfig = LLMConfig(
        model: LLMModel.gemma3_4b_q5_mm, // Multimodal model
        contextSize: 8096,
        nPredict: -1,
        nBatch: 8096,
        nThreads: 8,
        temperature: 0.7,
        topK: 64,
        topP: 0.95,
        minP: 0.05,
        penaltyRepeat: 1.1,
      );

      await FlutterLocalLlm.createWithDependencies(
        config: multimodalConfig,
        modelManager: mockModelManager,
        chatStorage: mockChatStorage,
        isolateManager: mockIsolateManager,
      );

      // Should download both text and image models (2 calls to getModelPath)
      verify(() => mockModelManager.getModelPath(any(), downloadUrl: any(named: 'downloadUrl'))).called(2);
    });
  });
}
