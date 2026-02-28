import 'package:flutter/material.dart';
import 'package:flutter_local_llm/flutter_local_llm.dart';
import '../widgets/chat_history_drawer.dart';
import '../widgets/loading_view.dart';
import '../widgets/model_picker.dart';

class ManualChatScreen extends StatefulWidget {
  const ManualChatScreen({super.key});

  @override
  State<ManualChatScreen> createState() => _ManualChatScreenState();
}

class _ManualChatScreenState extends State<ManualChatScreen> {
  // ChatManager is held here so chat history persists across model switches.
  final _chatManager = ChatManager();

  FlutterLocalLlm? _llm;
  LlmModel _currentModel = LlmModel.gemma3_1b_q5;
  bool _isLoading = true;
  double? _downloadProgress;
  bool _isGenerating = false;
  String? _error;

  String _pendingUserMessage = '';
  String _streamingResponse = '';

  final _controller = TextEditingController();
  final _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    _initializeLlm(_currentModel);
  }

  LlmConfig _configFor(LlmModel model) => switch (model) {
        LlmModel.gemma3_1b_q5 => LlmConfig.gemma3_1b_q5,
        LlmModel.gemma3n_E2B_q4 => LlmConfig.gemma3n_E2B_q4,
        LlmModel.gemma3_4b_q5_mm => LlmConfig.gemma3_4b_q5_mm,
        LlmModel.gemma3_4b_q3_mm => LlmConfig.gemma3_4b_q3_mm,
      };

  Future<void> _initializeLlm(LlmModel model) async {
    if (_isGenerating) return;
    setState(() {
      _isLoading = true;
      _downloadProgress = null;
      _error = null;
    });
    _llm?.dispose();
    _llm = null;

    try {
      _llm = await FlutterLocalLlm.createCustom(
        config: _configFor(model),
        chatManager: _chatManager,
        modelManager: ModelManager(
          onDownloadProgress: (p) {
            if (mounted) setState(() => _downloadProgress = p);
          },
        ),
      );
      if (mounted) setState(() { _currentModel = model; _isLoading = false; });
    } catch (e) {
      if (mounted) setState(() { _error = e.toString(); _isLoading = false; });
    }
  }

  Future<void> _sendMessage(String text) async {
    text = text.trim();
    if (text.isEmpty || _llm == null || _isGenerating) return;
    _controller.clear();
    setState(() {
      _isGenerating = true;
      _pendingUserMessage = text;
      _streamingResponse = '';
    });

    try {
      await for (final token in _llm!.sendMessage(text)) {
        if (mounted) {
          setState(() => _streamingResponse += token);
          _scrollToBottom();
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isGenerating = false;
          _pendingUserMessage = '';
          _streamingResponse = '';
        });
        _scrollToBottom();
      }
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _newChat() {
    if (_llm == null) return;
    _llm!.chatManager.startNewChat(systemPrompt: _llm!.config.systemPrompt);
    _llm!.chatManager.saveChats();
    setState(() {});
  }

  void _selectChat(int index) {
    _llm?.chatManager.activeChatIndex = index;
    setState(() {});
  }

  Future<void> _deleteChat(int index) async {
    await _llm?.chatManager.deleteChat(index);
    setState(() {});
  }

  @override
  void dispose() {
    _llm?.dispose();
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final llm = _llm;
    final status =
        _downloadProgress != null ? 'Downloading model...' : 'Loading model...';

    Widget body;
    if (_isLoading) {
      body = LoadingView(status: status, progress: _downloadProgress);
    } else if (_error != null) {
      body = Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.error_outline, size: 48),
              const SizedBox(height: 16),
              Text(_error!, textAlign: TextAlign.center),
              const SizedBox(height: 16),
              FilledButton(
                onPressed: () => _initializeLlm(_currentModel),
                child: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    } else {
      final messages = llm!.chatManager.activeChat.fullHistory
          .where((m) => m.role != Role.system)
          .toList();
      body = Column(
        children: [
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              itemCount: messages.length + (_isGenerating ? 2 : 0),
              itemBuilder: (context, index) {
                if (index < messages.length) {
                  final msg = messages[index];
                  return _MessageBubble(
                    text: msg.content,
                    isUser: msg.role == Role.user,
                  );
                }
                if (index == messages.length) {
                  return _MessageBubble(text: _pendingUserMessage, isUser: true);
                }
                return _MessageBubble(
                  text: _streamingResponse.isEmpty ? '…' : _streamingResponse,
                  isUser: false,
                );
              },
            ),
          ),
          if (_isGenerating) const LinearProgressIndicator(),
          _InputBar(
            controller: _controller,
            enabled: !_isGenerating,
            onSend: _sendMessage,
          ),
        ],
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: Text(
          llm != null ? llm.chatManager.activeChat.title : 'Manual Chat',
          overflow: TextOverflow.ellipsis,
        ),
        actions: llm != null
            ? [
                ModelPicker(
                  current: _currentModel,
                  enabled: !_isGenerating,
                  onSelected: _initializeLlm,
                ),
              ]
            : null,
      ),
      drawer: llm != null
          ? ChatHistoryDrawer(
              chats: llm.chatManager.chats,
              activeIndex: llm.chatManager.activeChatIndex,
              onSelectChat: _selectChat,
              onNewChat: _newChat,
              onDeleteChat: _deleteChat,
            )
          : null,
      body: body,
    );
  }
}

class _MessageBubble extends StatelessWidget {
  final String text;
  final bool isUser;

  const _MessageBubble({required this.text, required this.isUser});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.75,
        ),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color: isUser
              ? colorScheme.primary
              : colorScheme.surfaceContainerHighest,
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(20),
            topRight: const Radius.circular(20),
            bottomLeft: Radius.circular(isUser ? 20 : 4),
            bottomRight: Radius.circular(isUser ? 4 : 20),
          ),
        ),
        child: Text(
          text,
          style: theme.textTheme.bodyMedium?.copyWith(
            color: isUser ? colorScheme.onPrimary : colorScheme.onSurface,
          ),
        ),
      ),
    );
  }
}

class _InputBar extends StatelessWidget {
  final TextEditingController controller;
  final bool enabled;
  final void Function(String) onSend;

  const _InputBar({
    required this.controller,
    required this.enabled,
    required this.onSend,
  });

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
        child: Row(
          children: [
            Expanded(
              child: TextField(
                controller: controller,
                enabled: enabled,
                maxLines: null,
                textInputAction: TextInputAction.send,
                decoration: InputDecoration(
                  hintText: 'Message…',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(24),
                  ),
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 10,
                  ),
                ),
                onSubmitted: onSend,
              ),
            ),
            const SizedBox(width: 8),
            IconButton.filled(
              onPressed: enabled ? () => onSend(controller.text) : null,
              icon: const Icon(Icons.send),
            ),
          ],
        ),
      ),
    );
  }
}
