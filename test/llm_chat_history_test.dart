import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_local_llm/src/llm_chat_history.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'test_helpers.dart';

void main() {
  group('LlmChatHistory toJson() / fromJson()', () {
    test('serializes and deserializes complete chat history', () {
      // Create history with title, timestamps, messages
      final original = LlmChatHistory(
        title: 'Test Chat',
        createdAt: DateTime(2024, 1, 15, 10, 30),
        updatedAt: DateTime(2024, 1, 15, 11, 45),
      );

      original.addMessage(role: Role.system, content: 'System prompt');
      original.addMessage(
          role: Role.user, content: 'Hello', images: ['/path/to/image.jpg']);
      original.addMessage(role: Role.assistant, content: 'Hi there!');

      // Serialize and deserialize
      final json = original.toJson();
      final restored = LlmChatHistory.fromJson(json);

      // Verify all fields
      expect(restored.title, 'Test Chat');
      expect(restored.createdAt, DateTime(2024, 1, 15, 10, 30));
      expect(restored.updatedAt, DateTime(2024, 1, 15, 11, 45));
      expect(restored.fullHistory.length, 3);
      expect(restored.messages.length, 3);
      expect(restored.fullHistory[1].images, ['/path/to/image.jpg']);
    });

    test('applies default values for null title and timestamps', () {
      final beforeCreate = DateTime.now();
      final chat = LlmChatHistory(); // All params null
      final afterCreate = DateTime.now();

      expect(chat.title, 'New Chat');
      expect(timestampsNear(chat.createdAt, beforeCreate), true);
      expect(timestampsNear(chat.createdAt, afterCreate), true);

      // Serialize and deserialize
      final json = chat.toJson();
      final restored = LlmChatHistory.fromJson(json);

      expect(restored.title, 'New Chat');
      expect(restored.createdAt, chat.createdAt);
    });

    test('maintains separate fullHistory and active messages after trimming',
        () {
      final chat = LlmChatHistory();

      // Add multiple message pairs
      chat.addMessage(role: Role.system, content: 'System');
      for (int i = 0; i < 5; i++) {
        chat.addMessage(role: Role.user, content: 'User $i');
        chat.addMessage(role: Role.assistant, content: 'Assistant $i');
      }

      // Trim to keep only 2 recent pairs
      chat.autoTrimForSpaceNoLlama(keepRecentPairs: 2);

      // fullHistory should have all 11 messages
      expect(chat.fullHistory.length, 11);
      // messages should have system + 4 recent (2 pairs)
      expect(chat.messages.length, 5);

      // Serialize and restore
      final json = chat.toJson();
      final restored = LlmChatHistory.fromJson(json);

      // Both should be preserved
      expect(restored.fullHistory.length, 11);
      expect(restored.messages.length, 5);
    });

    test('restores contextStartIndex after deserialization', () {
      final chat = LlmChatHistory();
      chat.addMessage(role: Role.system, content: 'System');
      for (int i = 0; i < 10; i++) {
        chat.addMessage(role: Role.user, content: 'User $i');
        chat.addMessage(role: Role.assistant, content: 'Assistant $i');
      }

      // Trim to 2 pairs
      chat.autoTrimForSpaceNoLlama(keepRecentPairs: 2);

      final json = chat.toJson();
      expect(json['contextStartIndex'],
          greaterThan(0)); // Should point to recent messages

      final restored = LlmChatHistory.fromJson(json);
      // Verify active messages match (system + recent pairs)
      expect(restored.messages.first.content, 'System');
      expect(restored.messages.last.content, 'Assistant 9');
    });
  });

  group('LlmChatHistory shouldTrimBeforePromptNoLlama()', () {
    test('triggers trimming at 4/5 (80%) capacity', () {
      final chat = LlmChatHistory();
      final contextSize = 10000;

      // Just below 20% remaining - should trigger
      expect(
        chat.shouldTrimBeforePromptNoLlama(
          'Short prompt',
          (contextSize ~/ 5) - 100, // 1900 remaining (19%)
          contextSize,
        ),
        true,
      );

      // At exactly 20% remaining - should NOT trigger (not less than)
      expect(
        chat.shouldTrimBeforePromptNoLlama(
          'Short prompt',
          contextSize ~/ 5, // 2000 remaining (20%)
          contextSize,
        ),
        false,
      );

      // Above threshold (21% remaining) with small prompt - should not trigger
      expect(
        chat.shouldTrimBeforePromptNoLlama(
          'Short prompt',
          (contextSize ~/ 5) + 100, // 2100 remaining (21%)
          contextSize,
        ),
        false,
      );
    });

    test('includes image tokens in trim decision', () {
      final chat = LlmChatHistory();
      final contextSize = 10000;
      final remainingSpace = 5000; // 50% remaining

      // 400 char prompt = ~100 tokens + 50 buffer = 150 tokens
      final prompt = 'x' * 400;

      // Without images: 150 < 5000 → no trim needed
      expect(
        chat.shouldTrimBeforePromptNoLlama(
          prompt,
          remainingSpace,
          contextSize,
          imageCount: 0,
        ),
        false,
      );

      // With 3 images: 150 + (3 * 300) = 1050 tokens < 5000 → no trim
      expect(
        chat.shouldTrimBeforePromptNoLlama(
          prompt,
          remainingSpace,
          contextSize,
          imageCount: 3,
        ),
        false,
      );

      // With 20 images: 150 + (20 * 300) = 6150 tokens > 5000 → trim needed
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
    test('preserves all system messages and specified number of recent pairs',
        () {
      final chat = LlmChatHistory();

      // Add 2 system messages
      chat.addMessage(role: Role.system, content: 'System 1');
      chat.addMessage(role: Role.system, content: 'System 2');

      // Add 10 user/assistant pairs
      for (int i = 0; i < 10; i++) {
        chat.addMessage(role: Role.user, content: 'User $i');
        chat.addMessage(role: Role.assistant, content: 'Assistant $i');
      }

      // fullHistory has 22 messages total
      expect(chat.fullHistory.length, 22);

      // Trim to keep 2 recent pairs
      chat.autoTrimForSpaceNoLlama(keepRecentPairs: 2);

      // messages should have: 2 system + 4 recent = 6
      expect(chat.messages.length, 6);

      // Verify system messages are first
      expect(chat.messages[0].content, 'System 1');
      expect(chat.messages[1].content, 'System 2');

      // Verify recent pairs are last
      expect(chat.messages[2].content, 'User 8');
      expect(chat.messages[3].content, 'Assistant 8');
      expect(chat.messages[4].content, 'User 9');
      expect(chat.messages[5].content, 'Assistant 9');

      // fullHistory unchanged
      expect(chat.fullHistory.length, 22);
    });

    test('does not crash when keepRecentPairs exceeds message count', () {
      final chat = LlmChatHistory();
      chat.addMessage(role: Role.system, content: 'System');
      chat.addMessage(role: Role.user, content: 'User 1');
      chat.addMessage(role: Role.assistant, content: 'Assistant 1');

      // Request more pairs than exist
      expect(
        () => chat.autoTrimForSpaceNoLlama(keepRecentPairs: 10),
        returnsNormally,
      );

      // All messages should be kept
      expect(chat.messages.length, 3);
      expect(chat.fullHistory.length, 3);
    });
  });
}
