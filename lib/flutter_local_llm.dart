import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
import 'flutter_local_llm_platform_interface.dart';

/// Supported models for FlutterLocalLLM
enum LLMModel { gemma3nE2B }

/// Extension to get model details
extension LLMModelExtension on LLMModel {
  String get name {
    switch (this) {
      case LLMModel.gemma3nE2B:
        return 'gemma-3n-E2B-it-Q4_K_M';
    }
  }

  String get url {
    switch (this) {
      case LLMModel.gemma3nE2B:
        return 'https://huggingface.co/unsloth/gemma-3n-E2B-it-GGUF/resolve/main/gemma-3n-E2B-it-Q4_K_M.gguf';
    }
  }

  String get fileName {
    return '$name.gguf';
  }
}

class FlutterLocalLlm {
  Future<String?> getPlatformVersion() {
    return FlutterLocalLlmPlatform.instance.getPlatformVersion();
  }

  /// Load a local LLM model
  ///
  /// [model] - The predefined model to load (required if customUrl is not provided)
  /// [customUrl] - Optional custom URL to download the GGUF file from
  /// [onDownloadProgress] - Optional callback for download progress (0.0 to 1.0)
  /// [contextParams] - Optional context parameters for the model
  /// [samplerParams] - Optional sampler parameters for the model
  ///
  /// Returns a [LlamaParent] instance that manages the loaded model
  Future<LlamaParent> loadModel({
    LLMModel? model,
    String? customUrl,
    void Function(double progress)? onDownloadProgress,
    ContextParams? contextParams,
    SamplerParams? samplerParams,
  }) async {
    if (kDebugMode) {
      print('[loadModel] Starting...');
    }

    // Validate inputs
    if (model == null && customUrl == null) {
      throw ArgumentError('Either model or customUrl must be provided');
    }

    // Determine download URL and file name
    final downloadUrl = customUrl ?? model!.url;
    final fileName = customUrl != null
        ? path.basename(Uri.parse(customUrl).path)
        : model!.fileName;

    if (kDebugMode) {
      print('[loadModel] Model: $fileName');
    }

    // Get models directory
    final modelsDir = await _getModelsDirectory();
    final modelFilePath = path.join(modelsDir.path, fileName);

    if (kDebugMode) {
      print('[loadModel] Model path: $modelFilePath');
    }

    // Download model if it doesn't exist
    if (!File(modelFilePath).existsSync()) {
      if (kDebugMode) {
        print('[loadModel] Downloading model from $downloadUrl...');
      }
      await _downloadModel(downloadUrl, modelFilePath, onDownloadProgress);
      if (kDebugMode) {
        print('[loadModel] Model downloaded to $modelFilePath');
      }
    } else {
      if (kDebugMode) {
        print('[loadModel] Model already exists at $modelFilePath');
      }
    }

    // Set up llama library path based on platform
    if (kDebugMode) {
      print('[loadModel] Setting up library path...');
    }
    _setupLlamaLibraryPath();

    // Set up default parameters if not provided
    final ctx = contextParams ?? _getDefaultContextParams();
    final sampler = samplerParams ?? _getDefaultSamplerParams();

    if (kDebugMode) {
      print('[loadModel] Creating LlamaLoad command...');
    }

    // Create load command
    final loadCommand = LlamaLoad(
      path: modelFilePath,
      modelParams: ModelParams(),
      contextParams: ctx,
      samplingParams: sampler,
    );

    if (kDebugMode) {
      print('[loadModel] Initializing LlamaParent...');
    }

    // Initialize LlamaParent
    final llamaParent = LlamaParent(loadCommand);

    if (kDebugMode) {
      print('[loadModel] Calling llamaParent.init()...');
    }

    await llamaParent.init();

    if (kDebugMode) {
      print(
        '[loadModel] init() completed. Current status: ${llamaParent.status}',
      );
    }

    // Wait for model to be ready
    int attempts = 0;
    const maxAttempts = 120; // 60 seconds timeout

    if (kDebugMode) {
      print('[loadModel] Waiting for model to be ready...');
    }

    while (llamaParent.status != LlamaStatus.ready && attempts < maxAttempts) {
      await Future.delayed(const Duration(milliseconds: 500));
      attempts++;

      if (attempts % 10 == 0 && kDebugMode) {
        print(
          '[loadModel] Still waiting... Status: ${llamaParent.status}, Attempt: $attempts',
        );
      }

      if (llamaParent.status == LlamaStatus.error) {
        if (kDebugMode) {
          print('[loadModel] ERROR: Model status is error');
        }
        throw Exception('Error loading model');
      }
    }

    if (attempts >= maxAttempts && llamaParent.status != LlamaStatus.ready) {
      if (kDebugMode) {
        print(
          '[loadModel] TIMEOUT: Model not ready after $maxAttempts attempts',
        );
      }
      throw TimeoutException(
        'Timeout waiting for model to be ready. Current status: ${llamaParent.status}',
      );
    }

    if (kDebugMode) {
      print(
        '[loadModel] Model loaded successfully! Status: ${llamaParent.status}',
      );
    }

    return llamaParent;
  }

  /// Get the models directory, creating it if it doesn't exist
  Future<Directory> _getModelsDirectory() async {
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
  Future<void> _downloadModel(
    String url,
    String destinationPath,
    void Function(double progress)? onProgress,
  ) async {
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
        onProgress(downloadedBytes / totalBytes);
      }
    }

    await sink.close();
  }

  /// Set up the llama library path based on the current platform
  void _setupLlamaLibraryPath() {
    // For iOS/macOS, the framework is embedded and linked via CocoaPods
    // FFI needs the library name that dyld can find via @rpath

    if (Platform.isMacOS) {
      // On macOS, use just the framework name - dyld will find it via @rpath
      // since it's embedded in the app bundle via CocoaPods
      // Llama.libraryPath = 'llama.framework/llama';
    } else if (Platform.isIOS) {
      // On iOS, same approach
      // Llama.libraryPath = 'llama.framework/llama';
    } else if (Platform.isAndroid) {
      // For Android, you would set the path to the .so file
      Llama.libraryPath = 'libllama.so';
    } else if (Platform.isLinux) {
      Llama.libraryPath = 'libllama.so';
    } else if (Platform.isWindows) {
      Llama.libraryPath = 'llama.dll';
    }
  }

  /// Get default context parameters
  ContextParams _getDefaultContextParams() {
    final params = ContextParams();
    params.nPredict = 8192;
    params.nCtx = 8192;
    params.nBatch = 512;
    return params;
  }

  /// Get default sampler parameters
  SamplerParams _getDefaultSamplerParams() {
    final params = SamplerParams();
    params.temp = 0.7;
    params.topK = 64;
    params.topP = 0.95;
    params.penaltyRepeat = 1.1;
    return params;
  }
}
