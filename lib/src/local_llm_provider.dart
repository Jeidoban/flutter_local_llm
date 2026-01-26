import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter_ai_toolkit/flutter_ai_toolkit.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'flutter_local_llm_base.dart';

class LocalLlmProvider extends LlmProvider with ChangeNotifier {
  final FlutterLocalLlm _llm;
  final List<ChatMessage> _chatHistory = [];

  /// Creates a LocalLlmProvider wrapping a FlutterLocalLlm instance
  /// The FlutterLocalLlm instance must be initialized before passing it here.
  /// Optionally provide [initialHistory] to restore a previous conversation.
  ///
  /// Example:
  /// ```dart
  /// final llm = await FlutterLocalLlm.init(
  ///   model: LLMModel.gemma3nE2B,
  ///   systemPrompt: 'You are helpful.',
  /// );
  /// final provider = LocalLlmProvider(llm: llm);
  /// ```
  LocalLlmProvider({
    required FlutterLocalLlm llm,
    Iterable<ChatMessage>? initialHistory,
  }) : _llm = llm {
    if (initialHistory != null) {
      _chatHistory.addAll(initialHistory);
    }
  }
  @override
  Stream<String> generateStream(
    String prompt, {
    Iterable<Attachment> attachments = const [],
  }) async* {
    yield* _llm.sendMessage(prompt, addToHistory: false);
  }

  @override
  Stream<String> sendMessageStream(
    String prompt, {
    Iterable<Attachment> attachments = const [],
  }) async* {
    // Create full prompt with attachments
    final fullPrompt = prompt + _attachmentsToText(attachments);

    // Create user message and add to history
    final userMessage = ChatMessage.user(fullPrompt, attachments.toList());
    _chatHistory.add(userMessage);

    // Create empty LLM message
    final llmMessage = ChatMessage.llm();
    _chatHistory.add(llmMessage);

    try {
      // Send message and stream tokens
      final buffer = StringBuffer();
      await for (final token in _llm.sendMessage(fullPrompt, role: Role.user)) {
        buffer.write(token);
        llmMessage.append(token);
        yield token;
      }

      // Notify listeners after completion
      notifyListeners();
    } catch (e) {
      // Remove incomplete messages from history on error
      if (_chatHistory.isNotEmpty && _chatHistory.last == llmMessage) {
        _chatHistory.removeLast();
      }
      if (_chatHistory.isNotEmpty && _chatHistory.last == userMessage) {
        _chatHistory.removeLast();
      }

      rethrow;
    }
  }

  @override
  Iterable<ChatMessage> get history {
    return _chatHistory;
  }

  @override
  set history(Iterable<ChatMessage> messages) {
    // Clear and update internal history
    _chatHistory.clear();
    _chatHistory.addAll(messages);

    // Sync to underlying LLM
    _syncHistoryToLlm();

    // Notify listeners
    notifyListeners();
  }

  /// Convert a flutter_ai_toolkit ChatMessage to llama_cpp_dart Message
  Message _chatToMessage(ChatMessage msg) {
    final role = msg.origin.isUser ? Role.user : Role.assistant;
    return Message(role: role, content: msg.text ?? '');
  }

  /// Convert attachments to text annotations
  ///
  /// Since the current implementation doesn't support multimodal input,
  /// this converts attachments to text descriptions that are appended
  /// to the prompt.
  ///
  /// File attachments show name and MIME type.
  /// Link attachments show the URL.
  String _attachmentsToText(Iterable<Attachment> attachments) {
    if (attachments.isEmpty) return '';

    final buffer = StringBuffer('\n\n[Attachments:\n');
    for (final attachment in attachments) {
      if (attachment is FileAttachment) {
        buffer.write('- File: ${attachment.name} (${attachment.mimeType})\n');
        if (kDebugMode) {
          print(
            '[LocalLlmProvider] File attachment not yet supported: ${attachment.name}',
          );
        }
      } else if (attachment is LinkAttachment) {
        buffer.write('- Link: ${attachment.url}\n');
      }
    }
    buffer.write(']');
    return buffer.toString();
  }

  /// Sync internal ChatMessage history to FlutterLocalLlm
  void _syncHistoryToLlm() {
    // Clear and rebuild
    _llm.clearHistory();

    // Manually re-add system prompt since clearHistory creates a fresh ChatHistory
    if (_llm.config.systemPrompt != null) {
      _llm.chatHistory.addMessage(
        role: Role.system,
        content: _llm.config.systemPrompt!,
      );
    }

    // Add all chat messages
    for (final chatMsg in _chatHistory) {
      final message = _chatToMessage(chatMsg);
      _llm.chatHistory.addMessage(role: message.role, content: message.content);
    }
  }

  @override
  void dispose() {
    _llm.dispose();
    super.dispose();
  }
}
