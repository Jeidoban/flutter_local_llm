import 'package:flutter_ai_toolkit/flutter_ai_toolkit.dart';
import 'package:flutter_local_llm/flutter_local_llm.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'test_helpers.dart';

class MockFlutterLocalLlm extends Mock implements FlutterLocalLlm {}

void main() {
  setUpAll(() {
    registerMocktailFallbacks();
  });

  group('LocalLlmProvider', () {
    late MockFlutterLocalLlm mockLlm;
    late MockChatManager mockChatManager;

    setUp(() {
      mockChatManager = MockChatManager();
      mockLlm = MockFlutterLocalLlm();

      when(() => mockLlm.chatManager).thenReturn(mockChatManager);
      when(() => mockChatManager.activeChat).thenReturn(LlmChatHistory());
      when(() => mockLlm.clearHistory()).thenAnswer((_) async {});
      when(() => mockLlm.dispose()).thenReturn(null);
      // Default: return empty stream for any sendMessage call
      when(
        () => mockLlm.sendMessage(
          any(),
          role: any(named: 'role'),
          addToHistory: any(named: 'addToHistory'),
          images: any(named: 'images'),
        ),
      ).thenAnswer((_) => Stream.fromIterable([]));
    });

    test('loads history from active chat, skipping system messages', () {
      final activeChat = LlmChatHistory();
      activeChat.addMessage(role: Role.system, content: 'System prompt');
      activeChat.addMessage(role: Role.user, content: 'Hello');
      activeChat.addMessage(role: Role.assistant, content: 'Hi there');
      when(() => mockChatManager.activeChat).thenReturn(activeChat);

      final provider = LocalLlmProvider(mockLlm);
      final history = provider.history.toList();

      expect(history.length, 2);
      expect(history[0].origin.isUser, true);
      expect(history[0].text, 'Hello');
      expect(history[1].origin.isUser, false);
    });

    test('empty active chat results in empty history', () {
      final provider = LocalLlmProvider(mockLlm);
      expect(provider.history.isEmpty, true);
    });

    test('generateStream yields tokens without modifying history', () async {
      when(
        () => mockLlm.sendMessage(
          any(),
          role: any(named: 'role'),
          addToHistory: any(named: 'addToHistory'),
          images: any(named: 'images'),
        ),
      ).thenAnswer((_) => Stream.fromIterable(['Hello', ' ', 'world']));

      final provider = LocalLlmProvider(mockLlm);

      final tokens = <String>[];
      await for (final token in provider.generateStream('Hi')) {
        tokens.add(token);
      }

      expect(tokens, ['Hello', ' ', 'world']);
      expect(provider.history.isEmpty, true);
      verify(
        () => mockLlm.sendMessage(
          any(),
          addToHistory: false,
          images: any(named: 'images'),
        ),
      ).called(1);
    });

    test('sendMessageStream adds user and llm messages to history', () async {
      when(
        () => mockLlm.sendMessage(
          any(),
          role: any(named: 'role'),
          addToHistory: any(named: 'addToHistory'),
          images: any(named: 'images'),
        ),
      ).thenAnswer((_) => Stream.fromIterable(['Hi', ' there']));

      final provider = LocalLlmProvider(mockLlm);

      final tokens = <String>[];
      await for (final token in provider.sendMessageStream('Hello')) {
        tokens.add(token);
      }

      expect(tokens, ['Hi', ' there']);
      final history = provider.history.toList();
      expect(history.length, 2);
      expect(history[0].origin.isUser, true);
      expect(history[0].text, 'Hello');
      expect(history[1].origin.isUser, false);
      expect(history[1].text, 'Hi there');
    });

    test('sendMessageStream notifies listeners on completion', () async {
      when(
        () => mockLlm.sendMessage(
          any(),
          role: any(named: 'role'),
          addToHistory: any(named: 'addToHistory'),
          images: any(named: 'images'),
        ),
      ).thenAnswer((_) => Stream.fromIterable(['Response']));

      final provider = LocalLlmProvider(mockLlm);

      var notified = false;
      provider.addListener(() => notified = true);

      await provider.sendMessageStream('Hello').drain<void>();

      expect(notified, true);
    });

    test('sendMessageStream rolls back history on error', () async {
      when(
        () => mockLlm.sendMessage(
          any(),
          role: any(named: 'role'),
          addToHistory: any(named: 'addToHistory'),
          images: any(named: 'images'),
        ),
      ).thenAnswer((_) => Stream.error(Exception('LLM error')));

      final provider = LocalLlmProvider(mockLlm);

      await expectLater(
        provider.sendMessageStream('Hello'),
        emitsError(isA<Exception>()),
      );

      expect(provider.history.isEmpty, true);
    });

    test('history setter replaces history and syncs to LLM', () async {
      final testChat = LlmChatHistory();
      when(() => mockChatManager.activeChat).thenReturn(testChat);

      final provider = LocalLlmProvider(mockLlm);

      final userMsg = ChatMessage.user('Hello', []);
      final llmMsg = ChatMessage.llm();
      llmMsg.append('Hi!');

      provider.history = [userMsg, llmMsg];
      await Future.delayed(Duration.zero);

      verify(() => mockLlm.clearHistory()).called(1);
      expect(provider.history.length, 2);
      expect(testChat.messages.length, 2);
      expect(testChat.messages[0].role, Role.user);
      expect(testChat.messages[0].content, 'Hello');
      expect(testChat.messages[1].role, Role.assistant);
    });

    test('reloadHistory clears and reloads from active chat and notifies', () {
      final provider = LocalLlmProvider(mockLlm);

      final updatedChat = LlmChatHistory();
      updatedChat.addMessage(role: Role.user, content: 'New message');
      updatedChat.addMessage(role: Role.assistant, content: 'New response');
      when(() => mockChatManager.activeChat).thenReturn(updatedChat);

      var notified = false;
      provider.addListener(() => notified = true);

      provider.reloadHistory();

      expect(provider.history.length, 2);
      expect(notified, true);
    });

    test('dispose calls llm.dispose', () {
      final provider = LocalLlmProvider(mockLlm);
      provider.dispose();
      verify(() => mockLlm.dispose()).called(1);
    });
  });
}
