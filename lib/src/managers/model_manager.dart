import 'dart:async';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import '../isolate/llm_isolate.dart';
import '../models/llm_config.dart';

/// Manages model file downloading and isolate creation.
///
/// Handles resolving model paths, downloading `.gguf` files from the network
/// on first use, and spawning the [LlmIsolate] that runs inference.
///
/// Pass [modelsPath] to override the default storage location
/// (`<app_documents>/flutter_local_llm/models/`).
/// Pass [onDownloadProgress] to receive progress updates during downloads.
class ModelManager {
  String? modelsPath;
  void Function(double progress)? onDownloadProgress;

  ModelManager({this.modelsPath, this.onDownloadProgress});

  /// Returns the directory where model files are stored, creating it if needed.
  ///
  /// Uses [modelsPath] if set, otherwise defaults to
  /// `<app_documents>/flutter_local_llm/models/`.
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

  /// Returns the full filesystem path for [modelName], downloading it first if absent.
  ///
  /// If the file doesn't exist and [downloadUrl] is provided, downloads the
  /// model and reports progress via [onDownloadProgress].
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

  /// Downloads a model file from [url] and yields progress values from 0.0 to 1.0.
  ///
  /// Streams the response body directly to disk to avoid holding the full file
  /// in memory. If the server doesn't report a content length, yields 1.0
  /// on completion.
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

  /// Returns all `.gguf` files present in the models directory.
  Future<List<File>> listModels() async {
    final modelsDir = await _getModelsDirectory();
    return modelsDir
        .listSync()
        .whereType<File>()
        .where((f) => f.path.endsWith('.gguf'))
        .toList();
  }

  /// Deletes the model file with [modelName] from the models directory.
  ///
  /// Does nothing if the file does not exist.
  Future<void> deleteModel(String modelName) async {
    final modelsDir = await _getModelsDirectory();
    final file = File(path.join(modelsDir.path, modelName));
    if (file.existsSync()) {
      await file.delete();
    }
  }

  /// Downloads all model files required by [config] and spawns an [LlmIsolate].
  ///
  /// Downloads the main text model and, for multimodal configs, the vision
  /// projector as well. Both are downloaded to [modelsPath] (or the default
  /// location) before the isolate is created.
  Future<LlmIsolate> createModelIsolate(LlmConfig config) async {
    final modelPath = await getModelPath(
      config.fileName,
      downloadUrl: config.url,
    );

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
