import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';

/// Get the models directory, creating it if it doesn't exist
Future<Directory> getModelsDirectory() async {
  final documentsDir = await getApplicationDocumentsDirectory();
  final modelsDir = Directory(
    path.join(documentsDir.path, 'flutter_local_llm_models'),
  );

  if (!modelsDir.existsSync()) {
    modelsDir.createSync(recursive: true);
  }

  return modelsDir;
}

/// Download a model file from a URL
Future<void> downloadModel(
  String url,
  String destinationPath,
  void Function(double progress)? onProgress,
) async {
  if (kDebugMode) {
    print('[downloadModel] Starting download from $url');
    print('[downloadModel] Destination: $destinationPath');
  }

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

    if (onProgress != null && totalBytes > 0) {
      final progress = downloadedBytes / totalBytes;
      onProgress(progress);

      if (kDebugMode && downloadedBytes % (totalBytes ~/ 20) < chunk.length) {
        print(
          '[downloadModel] Progress: ${(progress * 100).toStringAsFixed(1)}%',
        );
      }
    }
  }

  await sink.close();

  if (kDebugMode) {
    print('[downloadModel] Download completed');
  }
}

/// Get the full path to a model file, downloading it if necessary
Future<String> getModelPath(
  String downloadUrl,
  String fileName,
  void Function(double progress)? onDownloadProgress,
) async {
  final modelsDir = await getModelsDirectory();
  final modelFilePath = path.join(modelsDir.path, fileName);

  if (!File(modelFilePath).existsSync()) {
    if (kDebugMode) {
      print('[getModelPath] Model not found, downloading...');
    }
    await downloadModel(downloadUrl, modelFilePath, onDownloadProgress);
    if (kDebugMode) {
      print('[getModelPath] Model downloaded to $modelFilePath');
    }
  } else {
    if (kDebugMode) {
      print('[getModelPath] Model already exists at $modelFilePath');
    }
  }

  return modelFilePath;
}
