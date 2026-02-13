import 'package:llama_cpp_dart/llama_cpp_dart.dart';

class LlmChatHistory extends ChatHistory {
  String title;
  DateTime createdAt;
  DateTime updatedAt;

  /// Estimated tokens per image for context calculations
  static const int tokensPerImage = 300;

  /// Index of the first message in the active context (from fullHistory)
  int _contextStartIndex = 0;

  LlmChatHistory({
    String? title,
    DateTime? createdAt,
    DateTime? updatedAt,
  })  : title = title ?? 'New Chat',
        createdAt = createdAt ?? DateTime.now(),
        updatedAt = updatedAt ?? DateTime.now();

  /// Automatically trims messages to fit within remaining space.
  /// Doesn't require Llama instance.
  /// Includes image token estimation when calculating space requirements.
  /// Rebuilds the active context from fullHistory, preserving full history.
  bool autoTrimForSpaceNoLlama({int keepRecentPairs = 2}) {
    // Extract system messages from fullHistory
    List<Message> systemMessages = fullHistory
        .where((msg) => msg.role == Role.system)
        .toList();

    // Collect recent message pairs from fullHistory
    List<Message> recentMessages = [];
    int pairsFound = 0;
    int recentStartIndex = fullHistory.length;

    for (
      int i = fullHistory.length - 1;
      i >= 0 && pairsFound < keepRecentPairs;
      i--
    ) {
      recentMessages.insert(0, fullHistory[i]);
      recentStartIndex = i;
      if (fullHistory[i].role == Role.user) pairsFound++;
    }

    // Track where recent messages start in fullHistory
    // System messages are always included from the beginning separately
    _contextStartIndex = recentStartIndex;

    // Rebuild messages (active context) from fullHistory
    messages.clear();
    messages.addAll(systemMessages);
    messages.addAll(recentMessages);

    return true;
  }

  /// Check if trimming is needed before adding a new prompt.
  /// Doesn't require Llama instance.
  /// Includes estimation for both text and attached images.
  /// Also triggers trimming if we're at 4/5 context capacity to leave room for response.
  bool shouldTrimBeforePromptNoLlama(
    String newPrompt,
    int remainingSpace,
    int contextSize, {
    int imageCount = 0,
  }) {
    // Trim if we're at 4/5 (80%) capacity to leave room for long responses
    // remainingSpace < contextSize / 5 means less than 20% is remaining
    if (remainingSpace < contextSize / 5) {
      return true;
    }

    // Estimate text tokens (~4 chars per token) + buffer
    int estimatedTextTokens = (newPrompt.length / 4).ceil() + 50;

    // Add estimated tokens for images
    int estimatedImageTokens = imageCount * tokensPerImage;

    int totalEstimatedTokens = estimatedTextTokens + estimatedImageTokens;

    return remainingSpace < totalEstimatedTokens;
  }

  @override
  Map<String, dynamic> toJson() {
    // Don't call super.toJson() - it only serializes messages (active context)
    // We want to serialize fullHistory (complete conversation)
    return {
      'title': title,
      'createdAt': createdAt.toIso8601String(),
      'updatedAt': updatedAt.toIso8601String(),
      'messages': fullHistory.map((m) => m.toJson()).toList(),
      'contextStartIndex': _contextStartIndex,
    };
  }

  factory LlmChatHistory.fromJson(Map<String, dynamic> json) {
    // Don't call ChatHistory.fromJson() - we need custom logic
    final history = LlmChatHistory(
      title: json['title'] as String?,
      createdAt: json['createdAt'] != null
          ? DateTime.parse(json['createdAt'] as String)
          : null,
      updatedAt: json['updatedAt'] != null
          ? DateTime.parse(json['updatedAt'] as String)
          : null,
    );

    // Load full history
    final messagesList = json['messages'] as List<dynamic>?;
    if (messagesList != null) {
      for (final msgJson in messagesList) {
        final msg = Message.fromJson(msgJson as Map<String, dynamic>);
        history.fullHistory.add(msg);
      }
    }

    // Restore context range
    history._contextStartIndex = json['contextStartIndex'] as int? ?? 0;

    // Rebuild active context (messages) from fullHistory
    // System messages are always included from the beginning
    final systemMessages = history.fullHistory
        .where((msg) => msg.role == Role.system)
        .toList();

    // Recent messages start at contextStartIndex
    final recentMessages = history.fullHistory
        .skip(history._contextStartIndex)
        .toList();

    // Stitch together: system messages + recent messages
    history.messages.clear();
    history.messages.addAll(systemMessages);
    history.messages.addAll(recentMessages);

    return history;
  }
}
