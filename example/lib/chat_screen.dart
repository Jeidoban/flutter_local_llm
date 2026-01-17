import 'package:flutter/material.dart';
import 'dart:async';

import 'package:flutter_local_llm/flutter_local_llm.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class ChatMessage {
  final String text;
  final bool isUser;
  final DateTime timestamp;

  ChatMessage({
    required this.text,
    required this.isUser,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();
}

class _ChatScreenState extends State<ChatScreen> {
  final _flutterLocalLlm = FlutterLocalLlm();
  final _messageController = TextEditingController();
  final _scrollController = ScrollController();
  final List<ChatMessage> _messages = [];

  LlamaParent? _llamaParent;
  ChatHistory? _chatHistory;
  bool _isLoading = true;
  bool _isProcessing = false;
  String _loadingStatus = 'Initializing...';
  double _downloadProgress = 0.0;
  String _currentResponse = '';
  Completer<void>? _completionDone;

  @override
  void initState() {
    super.initState();
    _initializeModel();
  }

  Future<void> _initializeModel() async {
    try {
      setState(() {
        _loadingStatus = 'Loading model...';
      });

      print('DEBUG: Starting model load...');
      _llamaParent = await _flutterLocalLlm.loadModel(
        model: LLMModel.gemma3nE2B,
        onDownloadProgress: (progress) {
          setState(() {
            _downloadProgress = progress;
            _loadingStatus = 'Downloading model: ${(progress * 100).toStringAsFixed(1)}%';
          });
          print('DEBUG: Download progress: ${(progress * 100).toStringAsFixed(1)}%');
        },
      );
      print('DEBUG: Model load returned, llamaParent is: ${_llamaParent != null ? "NOT NULL" : "NULL"}');

      // Initialize chat history with system prompt
      _chatHistory = ChatHistory();
      _chatHistory!.addMessage(
        role: Role.system,
        content: 'You are a helpful, concise assistant. Keep your answers informative but brief.',
      );

      // Listen to token stream
      _llamaParent!.stream.listen((token) {
        setState(() {
          _currentResponse += token;
          // Update the last message if it's from assistant
          if (_messages.isNotEmpty && !_messages.last.isUser) {
            _messages[_messages.length - 1] = ChatMessage(
              text: _currentResponse,
              isUser: false,
            );
          }
        });
        _scrollToBottom();
      });

      // Listen for completion events
      _llamaParent!.completions.listen((event) {
        if (event.success) {
          // Update chat history with complete response
          if (_chatHistory!.messages.isNotEmpty &&
              _chatHistory!.messages.last.role == Role.assistant) {
            _chatHistory!.messages.last = Message(
              role: Role.assistant,
              content: _currentResponse,
            );
          }
          _currentResponse = '';
          if (_completionDone != null && !_completionDone!.isCompleted) {
            _completionDone!.complete();
          }
          setState(() {
            _isProcessing = false;
          });
        }
      });

      setState(() {
        _isLoading = false;
        _loadingStatus = 'Model loaded successfully!';
      });

      // Add welcome message
      _addMessage('Hello! I\'m ready to chat. How can I help you today?', isUser: false);
    } catch (e) {
      setState(() {
        _isLoading = false;
        _loadingStatus = 'Error loading model: $e';
      });
    }
  }

  void _addMessage(String text, {required bool isUser}) {
    setState(() {
      _messages.add(ChatMessage(text: text, isUser: isUser));
    });
    _scrollToBottom();
  }

  void _scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 100), () {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _sendMessage() async {
    if (_messageController.text.trim().isEmpty || _isProcessing || _llamaParent == null) {
      return;
    }

    final userMessage = _messageController.text.trim();
    _messageController.clear();

    // Add user message to UI
    _addMessage(userMessage, isUser: true);

    // Add to chat history
    _chatHistory!.addMessage(role: Role.user, content: userMessage);
    _chatHistory!.addMessage(role: Role.assistant, content: '');

    setState(() {
      _isProcessing = true;
      _currentResponse = '';
    });

    // Add empty assistant message that will be updated with streaming response
    _addMessage('', isUser: false);

    // Create completer for this message
    _completionDone = Completer<void>();

    // Generate prompt and send to model
    String prompt = _chatHistory!.exportFormat(
      ChatFormat.gemma,
      leaveLastAssistantOpen: true,
    );

    await _llamaParent!.sendPrompt(prompt);

    // Wait for completion
    try {
      await _completionDone!.future.timeout(
        const Duration(seconds: 120),
        onTimeout: () {
          setState(() {
            _isProcessing = false;
          });
        },
      );
    } catch (e) {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  @override
  void dispose() {
    _messageController.dispose();
    _scrollController.dispose();
    _llamaParent?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Flutter Local LLM Chat'),
        elevation: 2,
      ),
      body: _isLoading
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const CircularProgressIndicator(),
                  const SizedBox(height: 16),
                  Text(_loadingStatus),
                  if (_downloadProgress > 0 && _downloadProgress < 1)
                    Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: LinearProgressIndicator(value: _downloadProgress),
                    ),
                ],
              ),
            )
          : Column(
              children: [
                Expanded(
                  child: _messages.isEmpty
                      ? const Center(
                          child: Text('No messages yet'),
                        )
                      : ListView.builder(
                          controller: _scrollController,
                          padding: const EdgeInsets.all(8),
                          itemCount: _messages.length,
                          itemBuilder: (context, index) {
                            final message = _messages[index];
                            return _ChatBubble(message: message);
                          },
                        ),
                ),
                if (_isProcessing)
                  const Padding(
                    padding: EdgeInsets.all(8.0),
                    child: LinearProgressIndicator(),
                  ),
                _buildMessageInput(),
              ],
            ),
    );
  }

  Widget _buildMessageInput() {
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.1),
            blurRadius: 4,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _messageController,
              decoration: const InputDecoration(
                hintText: 'Type a message...',
                border: OutlineInputBorder(),
                contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              ),
              maxLines: null,
              textInputAction: TextInputAction.send,
              onSubmitted: (_) => _sendMessage(),
              enabled: !_isProcessing && !_isLoading,
            ),
          ),
          const SizedBox(width: 8),
          IconButton.filled(
            onPressed: _isProcessing || _isLoading ? null : _sendMessage,
            icon: const Icon(Icons.send),
          ),
        ],
      ),
    );
  }
}

class _ChatBubble extends StatelessWidget {
  final ChatMessage message;

  const _ChatBubble({required this.message});

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: message.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4, horizontal: 8),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        decoration: BoxDecoration(
          color: message.isUser
              ? Theme.of(context).colorScheme.primary
              : Theme.of(context).colorScheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(20),
        ),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.75,
        ),
        child: Text(
          message.text,
          style: TextStyle(
            color: message.isUser
                ? Theme.of(context).colorScheme.onPrimary
                : Theme.of(context).colorScheme.onSurface,
          ),
        ),
      ),
    );
  }
}
