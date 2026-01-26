import 'package:llama_cpp_dart/llama_cpp_dart.dart';

class LlmChatHistory extends ChatHistory {
  LlmChatHistory({super.keepRecentPairs});

  /// Automatically trims messages to fit within remaining space.
  /// Doesn't require Llama instance.
  bool autoTrimForSpaceNoLlama(int remainingSpace, {int reserveTokens = 100}) {
    if (remainingSpace > reserveTokens) return false;

    List<Message> systemMessages = messages
        .where((msg) => msg.role == Role.system)
        .toList();
    List<Message> recentMessages = [];
    int pairsFound = 0;
    for (
      int i = messages.length - 1;
      i >= 0 && pairsFound < keepRecentPairs;
      i--
    ) {
      recentMessages.insert(0, messages[i]);
      if (messages[i].role == Role.user) pairsFound++;
    }

    messages.clear();
    messages.addAll(systemMessages);
    messages.addAll(recentMessages);
    return true;
  }

  /// Check if trimming is needed before adding a new prompt.
  /// Doesn't require Llama instance.
  bool shouldTrimBeforePromptNoLlama(int remainingSpace, String newPrompt) {
    int estimatedTokens = (newPrompt.length / 4).ceil() + 50;
    return remainingSpace < estimatedTokens;
  }
}
