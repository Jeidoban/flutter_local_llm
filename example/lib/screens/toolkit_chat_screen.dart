import 'package:flutter/material.dart';
import 'package:flutter_ai_toolkit/flutter_ai_toolkit.dart';
import 'package:flutter_local_llm/flutter_local_llm.dart';
import '../widgets/chat_history_drawer.dart';
import '../widgets/loading_view.dart';
import '../widgets/model_picker.dart';

class ToolkitChatScreen extends StatefulWidget {
  const ToolkitChatScreen({super.key});

  @override
  State<ToolkitChatScreen> createState() => _ToolkitChatScreenState();
}

class _ToolkitChatScreenState extends State<ToolkitChatScreen> {
  LocalLlmProvider? _provider;
  LlmModel _currentModel = LlmModel.gemma3_1b_q5;
  bool _isLoading = true;
  double? _downloadProgress;
  String? _error;

  @override
  void initState() {
    super.initState();
    _initializeLlm(_currentModel);
  }

  Future<void> _initializeLlm(LlmModel model) async {
    setState(() {
      _isLoading = true;
      _downloadProgress = null;
      _error = null;
    });

    _provider?.dispose();
    _provider = null;

    try {
      _provider = LocalLlmProvider(
        await FlutterLocalLlm.create(
          model: model,
          onDownloadProgress: (p) {
            if (mounted) setState(() => _downloadProgress = p);
          },
        ),
      );
      if (mounted) {
        setState(() {
          _currentModel = model;
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = e.toString();
          _isLoading = false;
        });
      }
    }
  }

  void _newChat() {
    if (_provider == null) return;
    final llm = _provider!.llm;
    llm.chatManager.startNewChat(systemPrompt: llm.config.systemPrompt);
    llm.chatManager.saveChats();
    _provider!.reloadHistory();
    setState(() {});
  }

  void _selectChat(int index) {
    if (_provider == null) return;
    _provider!.llm.chatManager.activeChatIndex = index;
    _provider!.reloadHistory();
    setState(() {});
  }

  Future<void> _deleteChat(int index) async {
    if (_provider == null) return;
    await _provider!.llm.chatManager.deleteChat(index);
    _provider!.reloadHistory();
    setState(() {});
  }

  @override
  void dispose() {
    _provider?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final provider = _provider;
    final status = _downloadProgress != null
        ? 'Downloading model...'
        : 'Loading model...';

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
      body = LlmChatView(provider: provider!);
    }

    final llm = provider?.llm;

    return Scaffold(
      appBar: AppBar(
        title: Text(
          llm != null ? llm.chatManager.activeChat.title : 'Toolkit Chat',
          overflow: TextOverflow.ellipsis,
        ),
        actions: llm != null
            ? [
                ModelPicker(
                  current: _currentModel,
                  enabled: true,
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
