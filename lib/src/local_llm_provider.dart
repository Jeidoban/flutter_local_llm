import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter_ai_toolkit/flutter_ai_toolkit.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'flutter_local_llm_base.dart';

class LocalLlmProvider extends LlmProvider with ChangeNotifier {
  final FlutterLocalLlm _llm;
  final List<ChatMessage> _chatHistory = [];

  /// Creates a LocalLlmProvider wrapping a FlutterLocalLlm instance
  /// The FlutterLocalLlm instance must be initialized before passing it here.
  /// Automatically loads the current active chat's history.
  ///
  /// Example:
  /// ```dart
  /// final llm = await FlutterLocalLlm.init(
  ///   model: LLMModel.gemma3nE2B,
  ///   systemPrompt: 'You are helpful.',
  /// );
  /// final provider = LocalLlmProvider(llm);
  /// ```
  LocalLlmProvider(FlutterLocalLlm llm) : _llm = llm {
    // Load existing chat history from active chat
    _loadHistoryFromActiveChat();
  }

  /// Load the active chat's history into the provider
  void _loadHistoryFromActiveChat() {
    final activeChat = _llm.activeChat;

    // Convert llama messages to ChatMessages
    for (final message in activeChat.messages) {
      // Skip system messages
      if (message.role == Role.system) continue;

      if (message.role == Role.user) {
        // Load attachments from image paths
        final attachments = <Attachment>[];
        if (message.images.isNotEmpty) {
          for (final imagePath in message.images) {
            final file = File(imagePath);
            if (file.existsSync()) {
              final bytes = file.readAsBytesSync();
              final name = file.uri.pathSegments.last;
              // Determine MIME type from file extension
              final mimeType = _getMimeType(name);
              attachments.add(
                ImageFileAttachment(
                  name: name,
                  bytes: bytes,
                  mimeType: mimeType,
                ),
              );
            }
          }
        }
        _chatHistory.add(ChatMessage.user(message.content, attachments));
      } else {
        final llmMessage = ChatMessage.llm();
        llmMessage.append(message.content);
        _chatHistory.add(llmMessage);
      }
    }
  }

  /// Reload history from the currently active chat
  /// Useful after switching chats in the underlying FlutterLocalLlm
  void reloadHistory() {
    _chatHistory.clear();
    _loadHistoryFromActiveChat();
    notifyListeners();
  }

  /// Determine MIME type from file extension
  String _getMimeType(String filename) {
    final ext = filename.toLowerCase().split('.').last;
    switch (ext) {
      case 'jpg':
      case 'jpeg':
        return 'image/jpeg';
      case 'png':
        return 'image/png';
      case 'gif':
        return 'image/gif';
      case 'webp':
        return 'image/webp';
      case 'bmp':
        return 'image/bmp';
      default:
        return 'image/jpeg'; // Default to JPEG
    }
  }

  @override
  Stream<String> generateStream(
    String prompt, {
    Iterable<Attachment> attachments = const [],
  }) async* {
    final attachmentFiles = await _extractAttachmentFiles(attachments);
    yield* _llm.sendMessage(
      prompt,
      addToHistory: false,
      images: attachmentFiles,
    );
  }

  @override
  Stream<String> sendMessageStream(
    String prompt, {
    Iterable<Attachment> attachments = const [],
  }) async* {
    // Extract attachment files from attachments
    final attachmentFiles = await _extractAttachmentFiles(attachments);

    // Create user message and add to history
    final userMessage = ChatMessage.user(prompt, attachments.toList());
    _chatHistory.add(userMessage);

    // Create empty LLM message
    final llmMessage = ChatMessage.llm();
    _chatHistory.add(llmMessage);

    try {
      // Send message and stream tokens with attachments
      final buffer = StringBuffer();
      await for (final token in _llm.sendMessage(
        prompt,
        role: Role.user,
        images: attachmentFiles,
      )) {
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

    // Sync to underlying LLM and notify listeners
    _syncHistoryToLlm().then((_) {
      notifyListeners();
    });
  }

  /// Convert a flutter_ai_toolkit ChatMessage to llama_cpp_dart Message
  Message _chatToMessage(ChatMessage msg) {
    final role = msg.origin.isUser ? Role.user : Role.assistant;
    return Message(role: role, content: msg.text ?? '');
  }

  /// Extract attachment files from attachments for multimodal input
  /// Writes attachment bytes to temporary files and returns the file list
  Future<List<File>?> _extractAttachmentFiles(
    Iterable<Attachment> attachments,
  ) async {
    final attachmentFiles = <File>[];

    for (final attachment in attachments) {
      if (attachment is ImageFileAttachment) {
        // Create temp file from bytes
        final tempDir = Directory.systemTemp;
        final timestamp = DateTime.now().millisecondsSinceEpoch;
        final tempFile = File(
          '${tempDir.path}/flutter_local_llm_${timestamp}_${attachment.name}',
        );
        await tempFile.writeAsBytes(attachment.bytes);
        attachmentFiles.add(tempFile);
      } else {
        throw UnsupportedError(
          'Unsupported attachment type: ${attachment.name}. Only images are supported.',
        );
      }
    }

    return attachmentFiles.isEmpty ? null : attachmentFiles;
  }

  /// Sync internal ChatMessage history to FlutterLocalLlm
  Future<void> _syncHistoryToLlm() async {
    // Clear and rebuild
    await _llm.clearHistory();

    // Get the active chat (clearHistory ensures one exists)
    final currentChat = _llm.activeChat;

    // Add all chat messages
    for (final chatMsg in _chatHistory) {
      final message = _chatToMessage(chatMsg);

      // Extract attachments if present
      List<String>? imagePaths;
      if (chatMsg.attachments.isNotEmpty) {
        final files = await _extractAttachmentFiles(chatMsg.attachments);
        imagePaths = files?.map((file) => file.path).toList();
      }

      currentChat.addMessage(
        role: message.role,
        content: message.content,
        images: imagePaths,
      );
    }
  }

  @override
  void dispose() {
    _llm.dispose();
    super.dispose();
  }
}
