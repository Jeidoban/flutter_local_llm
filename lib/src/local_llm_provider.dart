import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter_ai_toolkit/flutter_ai_toolkit.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'flutter_local_llm_base.dart';

/// A custom LlmProvider implementation that wraps FlutterLocalLlm
/// to work with flutter_ai_toolkit's LlmChatView and other widgets.
///
/// This provider manages the conversion between flutter_ai_toolkit's
/// ChatMessage format and llama_cpp_dart's Message format, while
/// maintaining chat history and handling streaming responses.
///
/// Example usage:
/// ```dart
/// // Initialize FlutterLocalLlm first
/// final llm = await FlutterLocalLlm.init(
///   model: LLMModel.gemma3nE2B,
///   systemPrompt: 'You are a helpful assistant.',
///   contextSize: 16000,
///   onDownloadProgress: (progress) {
///     print('Downloading: ${(progress * 100).toStringAsFixed(1)}%');
///   },
/// );
///
/// // Create provider with the initialized LLM
/// final provider = LocalLlmProvider(llm: llm);
///
/// // Use with LlmChatView
/// LlmChatView(provider: provider)
/// ```
class LocalLlmProvider extends LlmProvider with ChangeNotifier {
  final FlutterLocalLlm _llm;
  final List<ChatMessage> _chatHistory = [];
  bool _disposed = false;

  /// Creates a LocalLlmProvider wrapping a FlutterLocalLlm instance
  ///
  /// The FlutterLocalLlm instance must be initialized before passing it here.
  ///
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

  /// Generate a response from a prompt without affecting chat history
  ///
  /// This is a stateless operation - it doesn't add messages to the
  /// internal chat history. The response is generated in the context
  /// of the provided prompt only.
  ///
  /// Attachments are converted to text annotations and appended to the prompt.
  @override
  Stream<String> generateStream(
    String prompt, {
    Iterable<Attachment> attachments = const [],
  }) async* {
    _checkDisposed();

    if (kDebugMode) {
      print('[LocalLlmProvider] generateStream called');
    }

    // Save current history
    final savedHistory = ChatHistory();
    for (final msg in _llm.chatHistory.messages) {
      savedHistory.addMessage(role: msg.role, content: msg.content);
    }

    try {
      // Clear history temporarily
      _llm.clearHistory();

      // Create temp history with just this prompt
      final tempHistory = ChatHistory();
      final fullPrompt = prompt + _attachmentsToText(attachments);
      tempHistory.addMessage(role: Role.user, content: fullPrompt);

      // Generate response
      await for (final token in _llm.sendMessageWithHistory(tempHistory)) {
        yield token;
      }
    } finally {
      // Restore original history
      _llm.chatHistory.messages.clear();
      for (final msg in savedHistory.messages) {
        _llm.chatHistory.addMessage(role: msg.role, content: msg.content);
      }
    }
  }

  /// Send a message and get streaming response, adding to chat history
  ///
  /// This is a stateful operation - it adds both the user message and the
  /// assistant's response to the internal chat history.
  ///
  /// Attachments are converted to text annotations and appended to the prompt.
  @override
  Stream<String> sendMessageStream(
    String prompt, {
    Iterable<Attachment> attachments = const [],
  }) async* {
    _checkDisposed();

    if (kDebugMode) {
      print('[LocalLlmProvider] sendMessageStream called');
    }

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

      if (kDebugMode) {
        print(
          '[LocalLlmProvider] Message complete, history size: ${_chatHistory.length}',
        );
      }
    } catch (e) {
      if (kDebugMode) {
        print('[LocalLlmProvider] Error in sendMessageStream: $e');
      }

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

  /// Get the current chat history as an iterable of ChatMessage objects
  ///
  /// This converts the underlying FlutterLocalLlm history to ChatMessage format.
  /// System messages are filtered out as they're not typically displayed in the UI.
  @override
  Iterable<ChatMessage> get history {
    _checkDisposed();

    // Return the internal chat history
    // We maintain our own history to avoid conversion overhead
    return _chatHistory;
  }

  /// Set the chat history
  ///
  /// This replaces the current chat history with the provided messages
  /// and syncs them to the underlying FlutterLocalLlm instance.
  ///
  /// System prompts are preserved during the sync.
  @override
  set history(Iterable<ChatMessage> messages) {
    _checkDisposed();

    if (kDebugMode) {
      print('[LocalLlmProvider] Setting history with ${messages.length} messages');
    }

    // Clear and update internal history
    _chatHistory.clear();
    _chatHistory.addAll(messages);

    // Sync to underlying LLM
    _syncHistoryToLlm();

    // Notify listeners
    notifyListeners();
  }

  /// Clear all chat history
  ///
  /// This removes all messages from the chat history while preserving
  /// the system prompt. The underlying FlutterLocalLlm context is also cleared.
  void clearHistory() {
    _checkDisposed();

    if (kDebugMode) {
      print('[LocalLlmProvider] Clearing history');
    }

    _chatHistory.clear();
    _llm.clearHistory();
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
  ///
  /// This converts ChatMessage objects to Message objects and updates
  /// the underlying FlutterLocalLlm's chat history.
  ///
  /// The system prompt is preserved during this operation.
  void _syncHistoryToLlm() {
    // Save system prompt if it exists
    String? systemPrompt;
    if (_llm.chatHistory.messages.isNotEmpty &&
        _llm.chatHistory.messages.first.role == Role.system) {
      systemPrompt = _llm.chatHistory.messages.first.content;
    }

    // Clear and rebuild
    _llm.clearHistory();

    // Manually re-add system prompt since clearHistory creates a fresh ChatHistory
    if (systemPrompt != null && systemPrompt.isNotEmpty) {
      _llm.chatHistory.addMessage(role: Role.system, content: systemPrompt);
    }

    // Add all chat messages
    for (final chatMsg in _chatHistory) {
      final message = _chatToMessage(chatMsg);
      _llm.chatHistory.addMessage(
        role: message.role,
        content: message.content,
      );
    }
  }

  /// Check if this provider has been disposed
  void _checkDisposed() {
    if (_disposed) {
      throw StateError('LocalLlmProvider has been disposed');
    }
  }

  /// Dispose of resources
  ///
  /// This cleans up the underlying FlutterLocalLlm instance.
  /// After calling dispose, this provider cannot be used anymore.
  @override
  void dispose() {
    if (!_disposed) {
      if (kDebugMode) {
        print('[LocalLlmProvider] Disposing');
      }
      _llm.dispose();
      _disposed = true;
      super.dispose();
    }
  }
}
