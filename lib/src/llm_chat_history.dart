import 'package:flutter_local_llm/flutter_local_llm.dart';

class LlmChatHistory extends ChatHistory {
  LlmChatHistory({super.keepRecentPairs});

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

  bool shouldTrimBeforePromptNoLlama(int remainingSpace, String newPrompt) {
    int estimatedTokens = (newPrompt.length / 4).ceil() + 50;
    return remainingSpace < estimatedTokens;
  }
}
