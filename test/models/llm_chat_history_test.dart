import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_local_llm/src/models/llm_chat_history.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import '../test_helpers.dart';

void main() {
  group('LlmChatHistory toJson() / fromJson()', () {
    test('serializes and deserializes complete chat history', () {
      final original = LlmChatHistory(
        title: 'Test Chat',
        createdAt: DateTime(2024, 1, 15, 10, 30),
        updatedAt: DateTime(2024, 1, 15, 11, 45),
      );

      original.addMessage(role: Role.system, content: 'System prompt');
      original.addMessage(
        role: Role.user,
        content: 'Hello',
        images: ['/path/to/image.jpg'],
      );
      original.addMessage(role: Role.assistant, content: 'Hi there!');

      final json = original.toJson();
      final restored = LlmChatHistory.fromJson(json);

      expect(restored.title, 'Test Chat');
      expect(restored.createdAt, DateTime(2024, 1, 15, 10, 30));
      expect(restored.updatedAt, DateTime(2024, 1, 15, 11, 45));
      expect(restored.fullHistory.length, 3);
      expect(restored.messages.length, 3);
      expect(restored.fullHistory[1].images, ['/path/to/image.jpg']);
    });

    test('applies default values for null title and timestamps', () {
      final beforeCreate = DateTime.now();
      final chat = LlmChatHistory();
      final afterCreate = DateTime.now();

      expect(chat.title, 'New Chat');
      expect(timestampsNear(chat.createdAt, beforeCreate), true);
      expect(timestampsNear(chat.createdAt, afterCreate), true);

      final json = chat.toJson();
      final restored = LlmChatHistory.fromJson(json);

      expect(restored.title, 'New Chat');
      expect(restored.createdAt, chat.createdAt);
    });

    test(
      'maintains separate fullHistory and active messages after trimming',
      () {
        final chat = LlmChatHistory();

        chat.addMessage(role: Role.system, content: 'System');
        for (int i = 0; i < 5; i++) {
          chat.addMessage(role: Role.user, content: 'User $i');
          chat.addMessage(role: Role.assistant, content: 'Assistant $i');
        }

        chat.autoTrimForSpaceNoLlama(keepRecentPairs: 2);

        expect(chat.fullHistory.length, 11);
        expect(chat.messages.length, 5);

        final json = chat.toJson();
        final restored = LlmChatHistory.fromJson(json);

        expect(restored.fullHistory.length, 11);
        expect(restored.messages.length, 5);
      },
    );

    test('restores contextStartIndex after deserialization', () {
      final chat = LlmChatHistory();
      chat.addMessage(role: Role.system, content: 'System');
      for (int i = 0; i < 10; i++) {
        chat.addMessage(role: Role.user, content: 'User $i');
        chat.addMessage(role: Role.assistant, content: 'Assistant $i');
      }

      chat.autoTrimForSpaceNoLlama(keepRecentPairs: 2);

      final json = chat.toJson();
      expect(json['contextStartIndex'], greaterThan(0));

      final restored = LlmChatHistory.fromJson(json);
      expect(restored.messages.first.content, 'System');
      expect(restored.messages.last.content, 'Assistant 9');
    });
  });

  group('LlmChatHistory shouldTrimBeforePromptNoLlama()', () {
    test('triggers trimming at 4/5 (80%) capacity', () {
      final chat = LlmChatHistory();
      final contextSize = 10000;

      expect(
        chat.shouldTrimBeforePromptNoLlama(
          'Short prompt',
          (contextSize ~/ 5) - 100,
          contextSize,
        ),
        true,
      );

      expect(
        chat.shouldTrimBeforePromptNoLlama(
          'Short prompt',
          contextSize ~/ 5,
          contextSize,
        ),
        false,
      );

      expect(
        chat.shouldTrimBeforePromptNoLlama(
          'Short prompt',
          (contextSize ~/ 5) + 100,
          contextSize,
        ),
        false,
      );
    });

    test('includes image tokens in trim decision', () {
      final chat = LlmChatHistory();
      final contextSize = 10000;
      final remainingSpace = 5000;
      final prompt = 'x' * 400;

      expect(
        chat.shouldTrimBeforePromptNoLlama(
          prompt,
          remainingSpace,
          contextSize,
          imageCount: 0,
        ),
        false,
      );

      expect(
        chat.shouldTrimBeforePromptNoLlama(
          prompt,
          remainingSpace,
          contextSize,
          imageCount: 3,
        ),
        false,
      );

      expect(
        chat.shouldTrimBeforePromptNoLlama(
          prompt,
          remainingSpace,
          contextSize,
          imageCount: 20,
        ),
        true,
      );
    });
  });

  group('LlmChatHistory autoTrimForSpaceNoLlama()', () {
    test(
      'preserves all system messages and specified number of recent pairs',
      () {
        final chat = LlmChatHistory();

        chat.addMessage(role: Role.system, content: 'System 1');
        chat.addMessage(role: Role.system, content: 'System 2');

        for (int i = 0; i < 10; i++) {
          chat.addMessage(role: Role.user, content: 'User $i');
          chat.addMessage(role: Role.assistant, content: 'Assistant $i');
        }

        expect(chat.fullHistory.length, 22);

        chat.autoTrimForSpaceNoLlama(keepRecentPairs: 2);

        expect(chat.messages.length, 6);
        expect(chat.messages[0].content, 'System 1');
        expect(chat.messages[1].content, 'System 2');
        expect(chat.messages[2].content, 'User 8');
        expect(chat.messages[3].content, 'Assistant 8');
        expect(chat.messages[4].content, 'User 9');
        expect(chat.messages[5].content, 'Assistant 9');
        expect(chat.fullHistory.length, 22);
      },
    );

    test('does not crash when keepRecentPairs exceeds message count', () {
      final chat = LlmChatHistory();
      chat.addMessage(role: Role.system, content: 'System');
      chat.addMessage(role: Role.user, content: 'User 1');
      chat.addMessage(role: Role.assistant, content: 'Assistant 1');

      expect(
        () => chat.autoTrimForSpaceNoLlama(keepRecentPairs: 10),
        returnsNormally,
      );

      expect(chat.messages.length, 3);
      expect(chat.fullHistory.length, 3);
    });
  });
}
