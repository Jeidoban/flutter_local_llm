import 'dart:async';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';

/// Manages model downloading and file access
class ModelManager {
  late Directory modelsPath;
  final void Function(double progress)? onDownloadProgress;

  ModelManager({String? modelsPath, this.onDownloadProgress}) {
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
  Future<String> getModelPath(String modelName, {String? downloadUrl}) async {
    final modelFilePath = path.join(modelsPath.path, modelName);

    if (!File(modelFilePath).existsSync() && downloadUrl != null) {
      await for (final progress in downloadModel(modelName, downloadUrl)) {
        onDownloadProgress?.call(progress);
      }
    }

    return modelFilePath;
  }

  /// Download a model file and yield progress updates (0.0 to 1.0)
  Stream<double> downloadModel(String modelName, String url) async* {
    final modelFilePath = path.join(modelsPath.path, modelName);

    final request = await http.Client().send(
      http.Request('GET', Uri.parse(url)),
    );
    final totalBytes = request.contentLength ?? 0;
    int downloadedBytes = 0;

    final file = File(modelFilePath);
    final sink = file.openWrite();

    await for (final chunk in request.stream) {
      sink.add(chunk);
      downloadedBytes += chunk.length;

      if (totalBytes > 0) {
        yield downloadedBytes / totalBytes;
      }
    }

    await sink.close();

    if (totalBytes == 0) yield 1.0;
  }
}
