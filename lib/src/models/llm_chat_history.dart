import 'package:llama_cpp_dart/llama_cpp_dart.dart';

/// A chat session with metadata and context window management.
///
/// Extends [ChatHistory] from llama_cpp_dart with a [title], timestamps, and
/// persistence support. Maintains two views of the conversation:
///
/// - [fullHistory] — the complete archive, every message ever sent.
/// - [messages] — the active context window, the subset that fits within the
///   model's token limit and is actually sent to the LLM.
///
/// When the context fills up, [autoTrimForSpaceNoLlama] rebuilds [messages]
/// from [fullHistory], keeping system messages and the most recent
/// [keepRecentPairs] user/assistant exchanges. No conversation data is lost.
class LlmChatHistory extends ChatHistory {
  String title;
  DateTime createdAt;
  DateTime updatedAt;

  /// Estimated tokens consumed per image attachment, used for context budget calculations.
  static const int tokensPerImage = 300;

  /// Index into [fullHistory] where the active context window begins.
  int _contextStartIndex = 0;

  LlmChatHistory({
    String? title,
    DateTime? createdAt,
    DateTime? updatedAt,
  })  : title = title ?? 'New Chat',
        createdAt = createdAt ?? DateTime.now(),
        updatedAt = updatedAt ?? DateTime.now();

  /// Rebuilds [messages] to keep only system messages and the [keepRecentPairs]
  /// most recent user/assistant pairs from [fullHistory].
  ///
  /// Called when the context is nearly full. Does not discard [fullHistory].
  void autoTrimForSpaceNoLlama({int keepRecentPairs = 2}) {
    List<Message> systemMessages = fullHistory
        .where((msg) => msg.role == Role.system)
        .toList();

    List<Message> recentMessages = [];
    int pairsFound = 0;
    int recentStartIndex = fullHistory.length;

    for (
      int i = fullHistory.length - 1;
      i >= 0 && pairsFound < keepRecentPairs;
      i--
    ) {
      if (fullHistory[i].role == Role.system) {
        continue;
      }
      recentMessages.insert(0, fullHistory[i]);
      recentStartIndex = i;
      if (fullHistory[i].role == Role.user) pairsFound++;
    }

    _contextStartIndex = recentStartIndex;

    messages.clear();
    messages.addAll(systemMessages);
    messages.addAll(recentMessages);
  }

  /// Returns true if the context needs to be rebuilt before sending [newPrompt].
  ///
  /// Triggers when remaining space drops below 20% of [contextSize], or when
  /// the estimated token cost of [newPrompt] plus [imageCount] attached images
  /// exceeds [remainingSpace]. Text is estimated at ~4 characters per token.
  bool shouldTrimBeforePromptNoLlama(
    String newPrompt,
    int remainingSpace,
    int contextSize, {
    int imageCount = 0,
  }) {
    if (remainingSpace < contextSize / 5) {
      return true;
    }

    int estimatedTextTokens = (newPrompt.length / 4).ceil() + 50;
    int estimatedImageTokens = imageCount * tokensPerImage;
    int totalEstimatedTokens = estimatedTextTokens + estimatedImageTokens;

    return remainingSpace < totalEstimatedTokens;
  }

  /// Serializes the full conversation history (not just the active context window).
  ///
  /// Overrides [ChatHistory.toJson], which only serializes [messages].
  /// Persists [fullHistory] and [_contextStartIndex] so the active context
  /// window can be reconstructed exactly on load.
  @override
  Map<String, dynamic> toJson() {
    return {
      'title': title,
      'createdAt': createdAt.toIso8601String(),
      'updatedAt': updatedAt.toIso8601String(),
      'messages': fullHistory.map((m) => m.toJson()).toList(),
      'contextStartIndex': _contextStartIndex,
    };
  }

  /// Restores a [LlmChatHistory] from JSON produced by [toJson].
  ///
  /// Reconstructs both [fullHistory] and the active context window ([messages])
  /// by stitching system messages together with recent messages starting at
  /// the saved [_contextStartIndex].
  factory LlmChatHistory.fromJson(Map<String, dynamic> json) {
    final history = LlmChatHistory(
      title: json['title'] as String?,
      createdAt: json['createdAt'] != null
          ? DateTime.parse(json['createdAt'] as String)
          : null,
      updatedAt: json['updatedAt'] != null
          ? DateTime.parse(json['updatedAt'] as String)
          : null,
    );

    final messagesList = json['messages'] as List<dynamic>?;
    if (messagesList != null) {
      for (final msgJson in messagesList) {
        final msg = Message.fromJson(msgJson as Map<String, dynamic>);
        history.fullHistory.add(msg);
      }
    }

    history._contextStartIndex = json['contextStartIndex'] as int? ?? 0;

    final systemMessages = history.fullHistory
        .where((msg) => msg.role == Role.system)
        .toList();

    final recentMessages = history.fullHistory
        .skip(history._contextStartIndex)
        .where((msg) => msg.role != Role.system)
        .toList();

    history.messages.clear();
    history.messages.addAll(systemMessages);
    history.messages.addAll(recentMessages);

    return history;
  }
}
