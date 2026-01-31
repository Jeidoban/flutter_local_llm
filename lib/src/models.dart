import 'package:llama_cpp_dart/llama_cpp_dart.dart';

/// Supported models for FlutterLocalLLM
// ignore: constant_identifier_names
enum LLMModel { gemma3n_E2BTextOnly, gemma3_4b }

/// Extension to get model details
extension LLMModelExtension on LLMModel {
  String get name {
    switch (this) {
      case LLMModel.gemma3n_E2BTextOnly:
        return 'gemma-3n-E2B-it-Q4_K_M';
      case LLMModel.gemma3_4b:
        return 'gemma-3-4b-it-Q5_K_M';
    }
  }

  String get url {
    switch (this) {
      case LLMModel.gemma3n_E2BTextOnly:
        return 'https://huggingface.co/unsloth/gemma-3n-E2B-it-GGUF/resolve/main/gemma-3n-E2B-it-Q4_K_M.gguf';
      case LLMModel.gemma3_4b:
        return 'https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q5_K_M.gguf';
    }
  }

  String get fileName {
    return '$name.gguf';
  }

  String? get imageUrl {
    switch (this) {
      case LLMModel.gemma3n_E2BTextOnly:
        return null; // Text-only model
      case LLMModel.gemma3_4b:
        return 'https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/mmproj-F16.gguf';
    }
  }

  String? get imageFileName {
    switch (this) {
      case LLMModel.gemma3n_E2BTextOnly:
        return null;
      case LLMModel.gemma3_4b:
        return '$name-mmproj-F16.gguf';
    }
  }

  ChatFormat get chatFormat {
    switch (this) {
      case LLMModel.gemma3n_E2BTextOnly:
      case LLMModel.gemma3_4b:
        return ChatFormat.gemma;
    }
  }
}

/// Configuration for LLM initialization
class LLMConfig {
  final LLMModel model;
  final String? customModelUrl;
  final String? customImageModelUrl;
  final String? systemPrompt;
  final int contextSize;
  final int nPredict;
  final int nBatch;
  final int nThreads;
  final double temperature;
  final int topK;
  final double topP;
  final double minP;
  final double penaltyRepeat;
  final ChatFormat chatFormat;

  LLMConfig({
    this.model = LLMModel.gemma3n_E2BTextOnly,
    this.customModelUrl,
    this.customImageModelUrl,
    this.systemPrompt,
    this.contextSize = 16384,
    this.nPredict = -1,
    this.nBatch = 2048,
    this.nThreads = 8,
    this.temperature = 0.7,
    this.topK = 64,
    this.topP = 0.95,
    this.minP = 0.05,
    this.penaltyRepeat = 1.1,
    ChatFormat? chatFormat,
  }) : chatFormat = chatFormat ?? model.chatFormat;

  String get downloadUrl => customModelUrl ?? model.url;
  String get fileName {
    if (customModelUrl != null) {
      return Uri.parse(customModelUrl!).pathSegments.last;
    }
    return model.fileName;
  }

  String? get imageDownloadUrl => customImageModelUrl ?? model.imageUrl;
  String? get imageFileName {
    if (customImageModelUrl != null) {
      return Uri.parse(customImageModelUrl!).pathSegments.last;
    }
    return model.imageFileName;
  }
}
