import 'package:flutter/material.dart';
import 'package:flutter_ai_toolkit/flutter_ai_toolkit.dart';
import 'package:flutter_local_llm/flutter_local_llm.dart';

/// Example screen demonstrating LocalLlmProvider with flutter_ai_toolkit's LlmChatView
///
/// This shows how to use the LocalLlmProvider to wrap FlutterLocalLlm
/// and use it with the pre-built LlmChatView widget from flutter_ai_toolkit.
///
/// Compared to the custom ChatScreen, this provides:
/// - Less code (no custom UI implementation needed)
/// - Built-in features from flutter_ai_toolkit (formatting, markdown, etc.)
/// - Consistent UX with other LLM providers
class AiToolkitChatScreen extends StatefulWidget {
  const AiToolkitChatScreen({super.key});

  @override
  State<AiToolkitChatScreen> createState() => _AiToolkitChatScreenState();
}

class _AiToolkitChatScreenState extends State<AiToolkitChatScreen> {
  LocalLlmProvider? _provider;
  bool _isLoading = true;
  String _loadingStatus = 'Initializing...';
  double _downloadProgress = 0.0;

  @override
  void initState() {
    super.initState();
    _initializeProvider();
  }

  Future<void> _initializeProvider() async {
    try {
      setState(() => _loadingStatus = 'Loading model...');

      print('DEBUG: Starting FlutterLocalLlm initialization...');

      final llm = await FlutterLocalLlm.init(
        model: LLMModel.gemma3nE2B,
        systemPrompt: 'You are a helpful, concise assistant.',
        contextSize: 256,
        onDownloadProgress: (progress) {
          setState(() {
            _downloadProgress = progress;
            _loadingStatus =
                'Downloading: ${(progress * 100).toStringAsFixed(1)}%';
          });
          print(
            'DEBUG: Download progress: ${(progress * 100).toStringAsFixed(1)}%',
          );
        },
      );

      print('DEBUG: Creating LocalLlmProvider...');

      final provider = LocalLlmProvider(llm: llm);

      print('DEBUG: LocalLlmProvider ready');

      setState(() {
        _provider = provider;
        _isLoading = false;
      });
    } catch (e) {
      print('DEBUG: Error initializing: $e');
      setState(() {
        _isLoading = false;
        _loadingStatus = 'Error: $e';
      });
    }
  }

  @override
  void dispose() {
    _provider?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AI Toolkit Chat'),
        elevation: 2,
        actions: [
          if (_provider != null)
            IconButton(
              icon: const Icon(Icons.delete_outline),
              onPressed: () {
                _provider?.clearHistory();
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Chat history cleared')),
                );
              },
              tooltip: 'Clear History',
            ),
        ],
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
                      padding: const EdgeInsets.all(16),
                      child: LinearProgressIndicator(value: _downloadProgress),
                    ),
                ],
              ),
            )
          : _provider == null
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.error_outline, size: 48),
                  const SizedBox(height: 16),
                  Text(_loadingStatus),
                  const SizedBox(height: 16),
                  ElevatedButton(
                    onPressed: () {
                      setState(() => _isLoading = true);
                      _initializeProvider();
                    },
                    child: const Text('Retry'),
                  ),
                ],
              ),
            )
          : LlmChatView(provider: _provider!),
    );
  }
}
