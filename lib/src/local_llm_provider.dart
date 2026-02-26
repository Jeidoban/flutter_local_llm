import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter_ai_toolkit/flutter_ai_toolkit.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'flutter_local_llm_base.dart';

/// flutter_ai_toolkit [LlmProvider] backed by a local [FlutterLocalLlm] instance.
///
/// Translates between the toolkit's [ChatMessage] format and the internal
/// message format. To use, create a [FlutterLocalLlm] first, then wrap it:
///
/// ```dart
/// final provider = LocalLlmProvider(
///   await FlutterLocalLlm.create(model: LlmModel.gemma3_1b_q5),
/// );
/// provider.dispose();
/// ```
class LocalLlmProvider extends LlmProvider with ChangeNotifier {
  final FlutterLocalLlm llm;
  final List<ChatMessage> _chatHistory = [];

  LocalLlmProvider(this.llm) {
    _loadHistoryFromActiveChat();
  }

  /// Populates [_chatHistory] from the active chat's full history.
  ///
  /// Skips system messages. For user messages with image attachments, reads
  /// the files from disk and creates [ImageFileAttachment] objects.
  void _loadHistoryFromActiveChat() {
    final activeChat = llm.chatManager.activeChat;

    for (final message in activeChat.fullHistory) {
      if (message.role == Role.system) continue;

      if (message.role == Role.user) {
        final attachments = <Attachment>[];
        if (message.images.isNotEmpty) {
          for (final imagePath in message.images) {
            final file = File(imagePath);
            if (file.existsSync()) {
              final bytes = file.readAsBytesSync();
              final name = file.uri.pathSegments.last;
              attachments.add(
                ImageFileAttachment(
                  name: name,
                  bytes: bytes,
                  mimeType: _getMimeType(name),
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

  /// Clears [_chatHistory] and reloads from the currently active chat.
  ///
  /// Call this after switching chats in the underlying [FlutterLocalLlm].
  void reloadHistory() {
    _chatHistory.clear();
    _loadHistoryFromActiveChat();
    notifyListeners();
  }

  /// Returns the MIME type for an image [filename] based on its extension.
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
        return 'image/jpeg';
    }
  }

  @override
  Stream<String> generateStream(
    String prompt, {
    Iterable<Attachment> attachments = const [],
  }) async* {
    final attachmentFiles = await _extractAttachmentFiles(attachments);
    yield* llm.sendMessage(
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
    final attachmentFiles = await _extractAttachmentFiles(attachments);

    final userMessage = ChatMessage.user(prompt, attachments.toList());
    _chatHistory.add(userMessage);

    final llmMessage = ChatMessage.llm();
    _chatHistory.add(llmMessage);

    try {
      await for (final token in llm.sendMessage(
        prompt,
        role: Role.user,
        images: attachmentFiles,
      )) {
        llmMessage.append(token);
        yield token;
      }

      notifyListeners();
    } catch (e) {
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
    _chatHistory.clear();
    _chatHistory.addAll(messages);

    _syncHistoryToLlm().then((_) {
      notifyListeners();
    });
  }

  /// Converts a toolkit [ChatMessage] to a llama_cpp_dart [Message].
  Message _chatToMessage(ChatMessage msg) {
    final role = msg.origin.isUser ? Role.user : Role.assistant;
    return Message(role: role, content: msg.text ?? '');
  }

  /// Writes attachment bytes to temporary files and returns the resulting [File] list.
  ///
  /// Only [ImageFileAttachment] is supported. Throws [UnsupportedError] for
  /// other attachment types.
  Future<List<File>?> _extractAttachmentFiles(
    Iterable<Attachment> attachments,
  ) async {
    final attachmentFiles = <File>[];

    for (final attachment in attachments) {
      if (attachment is ImageFileAttachment) {
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

  /// Rebuilds the [FlutterLocalLlm] chat history from the current [_chatHistory].
  ///
  /// Clears the LLM context and re-adds all messages, writing image
  /// attachments to temp files for multimodal messages.
  Future<void> _syncHistoryToLlm() async {
    await llm.clearHistory();

    final currentChat = llm.chatManager.activeChat;

    for (final chatMsg in _chatHistory) {
      final message = _chatToMessage(chatMsg);

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
    llm.dispose();
    super.dispose();
  }
}
