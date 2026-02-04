import 'package:flutter/material.dart';
import 'package:flutter_ai_toolkit/flutter_ai_toolkit.dart';
import 'package:flutter_local_llm/flutter_local_llm.dart';

/// Example screen demonstrating LocalLlmProvider with flutter_ai_toolkit's LlmChatView
///
/// This shows how to use the LocalLlmProvider to wrap FlutterLocalLlm
/// and use it with the pre-built LlmChatView widget from flutter_ai_toolkit.
class AiToolkitChatScreen extends StatefulWidget {
  const AiToolkitChatScreen({super.key});

  @override
  State<AiToolkitChatScreen> createState() => _AiToolkitChatScreenState();
}

class _AiToolkitChatScreenState extends State<AiToolkitChatScreen> {
  LocalLlmProvider? _provider;
  bool _isLoading = true;
  String _loadingStatus = 'Initializing...';
  double _modelDownloadProgress = 0.0;
  double _imageModelDownloadProgress = 0.0;

  @override
  void initState() {
    super.initState();
    _initializeProvider();
  }

  Future<void> _initializeProvider() async {
    try {
      setState(() => _loadingStatus = 'Loading model...');

      final llm = await FlutterLocalLlm.init(
        model: LLMModel.gemma3_4b_q5_mm,
        systemPrompt: 'You are a helpful, concise assistant.',
        onModelDownloadProgress: (progress) {
          setState(() {
            _modelDownloadProgress = progress;
            _loadingStatus =
                'Downloading model: ${(progress * 100).toStringAsFixed(1)}%';
          });
        },
        onImageModelDownloadProgress: (progress) {
          setState(() {
            _imageModelDownloadProgress = progress;
            _loadingStatus =
                'Downloading image model: ${(progress * 100).toStringAsFixed(1)}%';
          });
        },
      );

      final provider = LocalLlmProvider(llm: llm);

      setState(() {
        _provider = provider;
        _isLoading = false;
      });
    } catch (e) {
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
                _provider?.history = [];
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
                  if (_modelDownloadProgress > 0 && _modelDownloadProgress < 1)
                    Padding(
                      padding: const EdgeInsets.all(16),
                      child: LinearProgressIndicator(
                        value: _modelDownloadProgress,
                      ),
                    ),
                  if (_imageModelDownloadProgress > 0 &&
                      _imageModelDownloadProgress < 1)
                    Padding(
                      padding: const EdgeInsets.all(16),
                      child: LinearProgressIndicator(
                        value: _imageModelDownloadProgress,
                      ),
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
