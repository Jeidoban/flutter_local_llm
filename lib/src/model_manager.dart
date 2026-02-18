import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';

/// Manages model downloading and file access
class ModelManager {
  late Directory modelsPath;

  ModelManager({String? modelsPath}) {
    if (modelsPath != null) {
      this.modelsPath = Directory(modelsPath);

      if (!this.modelsPath.existsSync()) {
        this.modelsPath.createSync(recursive: true);
      }
    } else {
      _getDefaultModelsDirectory().then((dir) => this.modelsPath = dir);
    }
  }

  Future<Directory> _getDefaultModelsDirectory() async {
    final documentsDir = await getApplicationDocumentsDirectory();
    final defaultModelsDir = Directory(
      path.join(documentsDir.path, 'flutter_local_llm', 'models'),
    );
    if (!defaultModelsDir.existsSync()) {
      defaultModelsDir.createSync(recursive: true);
    }
    return defaultModelsDir;
  }

  /// Get the full path to a model file, downloading if necessary
  Future<String> getModelPath(String downloadUrl, String modelName) async {
    final modelFilePath = path.join(modelsPath.path, '$modelName.gguf');

    if (!File(modelFilePath).existsSync()) {
      await _downloadModel(downloadUrl, modelFilePath);
    }

    return modelFilePath;
  }

  /// Download a model file from a URL
  Future<void> _downloadModel(String url, String destinationPath) async {
    final request = await http.Client().send(
      http.Request('GET', Uri.parse(url)),
    );
    final totalBytes = request.contentLength ?? 0;
    int downloadedBytes = 0;

    final file = File(destinationPath);
    final sink = file.openWrite();

    await for (final chunk in request.stream) {
      sink.add(chunk);
      downloadedBytes += chunk.length;

      if (onDownloadProgress != null && totalBytes > 0) {
        final progress = downloadedBytes / totalBytes;
        onDownloadProgress!(progress);
      }
    }

    await sink.close();
  }
}
