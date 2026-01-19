import 'package:llama_cpp_dart/llama_cpp_dart.dart';

/// Supported models for FlutterLocalLLM
enum LLMModel {
  gemma3nE2B,
}

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

  ChatFormat get chatFormat {
    switch (this) {
      case LLMModel.gemma3nE2B:
        return ChatFormat.gemma;
    }
  }
}

/// Configuration for LLM initialization
class LLMConfig {
  final LLMModel model;
  final String? customUrl;
  final String? systemPrompt;
  final int? contextSize;
  final int? nPredict;
  final int? nBatch;
  final double? temperature;
  final int? topK;
  final double? topP;
  final double? minP;
  final double? penaltyRepeat;
  final int keepRecentPairs;
  final ChatFormat chatFormat;

  LLMConfig({
    this.model = LLMModel.gemma3nE2B,
    this.customUrl,
    this.systemPrompt,
    this.contextSize,
    this.nPredict,
    this.nBatch,
    this.temperature = 0.7,
    this.topK = 64,
    this.topP = 0.95,
    this.minP = 0.05,
    this.penaltyRepeat = 1.1,
    this.keepRecentPairs = 2,
    ChatFormat? chatFormat,
  }) : chatFormat = chatFormat ?? model.chatFormat;

  String get downloadUrl => customUrl ?? model.url;
  String get fileName {
    if (customUrl != null) {
      return Uri.parse(customUrl!).pathSegments.last;
    }
    return model.fileName;
  }
}
