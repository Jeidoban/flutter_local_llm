import 'dart:async';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import '../isolate/llm_isolate.dart';
import '../models/llm_config.dart';

/// Manages model downloading and file access
class ModelManager {
  String? modelsPath;
  void Function(double progress)? onDownloadProgress;

  ModelManager({this.modelsPath, this.onDownloadProgress});

  Future<Directory> _getModelsDirectory() async {
    if (modelsPath != null) {
      final dir = Directory(modelsPath!);
      if (!dir.existsSync()) {
        dir.createSync(recursive: true);
      }
      return dir;
    }
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
    final modelsDir = await _getModelsDirectory();
    final modelFilePath = path.join(modelsDir.path, modelName);

    if (!File(modelFilePath).existsSync() && downloadUrl != null) {
      await for (final progress in downloadModel(modelName, downloadUrl)) {
        onDownloadProgress?.call(progress);
      }
    }

    return modelFilePath;
  }

  /// Download a model file and yield progress updates (0.0 to 1.0)
  Stream<double> downloadModel(String modelName, String url) async* {
    final modelsDir = await _getModelsDirectory();
    final modelFilePath = path.join(modelsDir.path, modelName);

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

  /// Create and initialize an isolate with the model
  ///
  /// Downloads models if needed and creates the isolate
  Future<LlmIsolate> createModelIsolate(LlmConfig config) async {
    // Download text model if needed
    final modelPath = await getModelPath(
      config.fileName,
      downloadUrl: config.url,
    );

    // Download image model if needed
    String? imageModelPath;
    if (config.imageUrl != null && config.imageFileName != null) {
      imageModelPath = await getModelPath(
        config.imageFileName!,
        downloadUrl: config.imageUrl!,
      );
    }

    return await LlmIsolate.spawn(
      modelPath,
      config,
      imageModelPath: imageModelPath,
    );
  }
}
