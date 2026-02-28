import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_local_llm/flutter_local_llm.dart';
import '../widgets/model_picker.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _modelManager = ModelManager();
  List<File> _downloadedFiles = [];
  bool _isLoadingModels = true;
  LlmModel? _downloadingModel;
  double _downloadProgress = 0;

  static const _configs = {
    LlmModel.gemma3_1b_q5: LlmConfig.gemma3_1b_q5,
    LlmModel.gemma3n_E2B_q4: LlmConfig.gemma3n_E2B_q4,
    LlmModel.gemma3_4b_q5_mm: LlmConfig.gemma3_4b_q5_mm,
    LlmModel.gemma3_4b_q3_mm: LlmConfig.gemma3_4b_q3_mm,
  };

  @override
  void initState() {
    super.initState();
    _loadDownloadedModels();
  }

  Future<void> _loadDownloadedModels() async {
    setState(() => _isLoadingModels = true);
    final files = await _modelManager.listModels();
    if (mounted) {
      setState(() {
        _downloadedFiles = files;
        _isLoadingModels = false;
      });
    }
  }

  bool _isDownloaded(LlmModel model) {
    final config = _configs[model]!;
    return _downloadedFiles
        .any((f) => f.uri.pathSegments.last == config.fileName);
  }

  Future<void> _downloadModel(LlmModel model) async {
    final config = _configs[model]!;
    setState(() {
      _downloadingModel = model;
      _downloadProgress = 0;
    });

    try {
      final hasImage =
          config.imageUrl != null && config.imageFileName != null;

      await for (final p
          in _modelManager.downloadModel(config.fileName, config.url)) {
        if (mounted) {
          setState(() => _downloadProgress = hasImage ? p * 0.5 : p);
        }
      }

      if (hasImage) {
        await for (final p in _modelManager.downloadModel(
          config.imageFileName!,
          config.imageUrl!,
        )) {
          if (mounted) {
            setState(() => _downloadProgress = 0.5 + p * 0.5);
          }
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Download failed: $e')),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _downloadingModel = null);
        await _loadDownloadedModels();
      }
    }
  }

  Future<void> _deleteModel(LlmModel model) async {
    final config = _configs[model]!;
    await _modelManager.deleteModel(config.fileName);
    if (config.imageFileName != null) {
      await _modelManager.deleteModel(config.imageFileName!);
    }
    await _loadDownloadedModels();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            tooltip: 'Refresh',
            onPressed: _loadDownloadedModels,
          ),
        ],
      ),
      body: _isLoadingModels
          ? const Center(child: CircularProgressIndicator())
          : ListView(
              padding: const EdgeInsets.all(16),
              children: [
                Text(
                  'Models',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 8),
                ...LlmModel.values.map(
                  (model) => _ModelTile(
                    model: model,
                    config: _configs[model]!,
                    isDownloaded: _isDownloaded(model),
                    isDownloading: _downloadingModel == model,
                    downloadProgress:
                        _downloadingModel == model ? _downloadProgress : null,
                    downloadedFiles: _downloadedFiles,
                    onDownload: _downloadingModel == null
                        ? () => _downloadModel(model)
                        : null,
                    onDelete: () => _deleteModel(model),
                  ),
                ),
              ],
            ),
    );
  }
}

class _ModelTile extends StatelessWidget {
  final LlmModel model;
  final LlmConfig config;
  final bool isDownloaded;
  final bool isDownloading;
  final double? downloadProgress;
  final List<File> downloadedFiles;
  final VoidCallback? onDownload;
  final VoidCallback onDelete;

  const _ModelTile({
    required this.model,
    required this.config,
    required this.isDownloaded,
    required this.isDownloading,
    required this.downloadProgress,
    required this.downloadedFiles,
    required this.onDownload,
    required this.onDelete,
  });

  String _totalSize() {
    final names = {
      config.fileName,
      if (config.imageFileName != null) config.imageFileName!,
    };
    final matched = downloadedFiles
        .where((f) => names.contains(f.uri.pathSegments.last));
    final bytes = matched.fold<int>(0, (sum, f) => sum + f.lengthSync());
    if (bytes == 0) return '';
    if (bytes < 1024 * 1024 * 1024) {
      return '${(bytes / (1024 * 1024)).toStringAsFixed(0)} MB';
    }
    return '${(bytes / (1024 * 1024 * 1024)).toStringAsFixed(2)} GB';
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isMultimodal = config.imageUrl != null;
    final sizeLabel = _totalSize();

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        ModelPicker.labelFor(model),
                        style: theme.textTheme.titleSmall,
                      ),
                      const SizedBox(height: 4),
                      Wrap(
                        spacing: 6,
                        children: [
                          if (isMultimodal)
                            Chip(
                              label: const Text('Multimodal'),
                              padding: EdgeInsets.zero,
                              labelPadding:
                                  const EdgeInsets.symmetric(horizontal: 6),
                              materialTapTargetSize:
                                  MaterialTapTargetSize.shrinkWrap,
                            ),
                          if (isDownloaded && sizeLabel.isNotEmpty)
                            Chip(
                              label: Text(sizeLabel),
                              padding: EdgeInsets.zero,
                              labelPadding:
                                  const EdgeInsets.symmetric(horizontal: 6),
                              materialTapTargetSize:
                                  MaterialTapTargetSize.shrinkWrap,
                            ),
                        ],
                      ),
                    ],
                  ),
                ),
                if (isDownloaded) ...[
                  Icon(Icons.check_circle,
                      color: theme.colorScheme.primary),
                  IconButton(
                    icon: const Icon(Icons.delete_outline),
                    tooltip: 'Delete',
                    onPressed: onDelete,
                  ),
                ] else if (isDownloading) ...[
                  const SizedBox(
                    width: 24,
                    height: 24,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
                ] else ...[
                  IconButton(
                    icon: const Icon(Icons.download_outlined),
                    tooltip: 'Download',
                    onPressed: onDownload,
                  ),
                ],
              ],
            ),
            if (isDownloading && downloadProgress != null) ...[
              const SizedBox(height: 8),
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: downloadProgress,
                  minHeight: 6,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                '${(downloadProgress! * 100).toStringAsFixed(0)}%',
                style: theme.textTheme.bodySmall,
              ),
            ],
          ],
        ),
      ),
    );
  }
}
