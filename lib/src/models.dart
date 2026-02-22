// ignore_for_file: non_constant_identifier_names, constant_identifier_names
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

/// Supported preset models
enum LlmModel { gemma3n_E2B_q4, gemma3_4b_q5_mm, gemma3_4b_q3_mm, gemma3_1b_q5 }

/// Self-contained LLM configuration — model identity, download info, and inference settings.
///
/// Use preset models via [LlmModel] enum with [FlutterLocalLlm.create], or
/// construct directly for custom models:
/// ```dart
/// LlmConfig(name: 'my-model', url: 'https://...', chatFormat: ChatFormat.chatml)
/// ```
class LlmConfig {
  final String name; // Base filename without .gguf
  final String url; // Download URL for the text model
  final String?
  imageUrl; // Download URL for vision projector (null = text-only)
  final ChatFormat chatFormat;
  final String systemPrompt;
  final int contextSize;
  final int nPredict;
  final int nBatch;
  final int nThreads;
  final double temperature;
  final int topK;
  final double topP;
  final double minP;
  final double penaltyRepeat;

  const LlmConfig({
    required this.name,
    required this.url,
    this.imageUrl,
    this.chatFormat = ChatFormat.gemma,
    this.systemPrompt = 'You are a helpful assistant.',
    this.contextSize = 16384,
    this.nPredict = -1,
    this.nBatch = 2048,
    this.nThreads = 8,
    this.temperature = 0.7,
    this.topK = 64,
    this.topP = 0.95,
    this.minP = 0.05,
    this.penaltyRepeat = 1.1,
  });

  // Derived filenames
  String get fileName => '$name.gguf';
  String? get imageFileName => imageUrl != null
      ? '$name-${Uri.parse(imageUrl!).pathSegments.last}'
      : null;

  // Preset models
  static const gemma3_1b_q5 = LlmConfig(
    name: 'gemma-3-1b-it-Q5_K_M',
    url:
        'https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q5_K_M.gguf',
  );

  static const gemma3n_E2B_q4 = LlmConfig(
    name: 'gemma-3n-E2B-it-Q4_K_M',
    url:
        'https://huggingface.co/unsloth/gemma-3n-E2B-it-GGUF/resolve/main/gemma-3n-E2B-it-Q4_K_M.gguf',
  );

  static const gemma3_4b_q5_mm = LlmConfig(
    name: 'gemma-3-4b-it-Q5_K_M',
    url:
        'https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q5_K_M.gguf',
    imageUrl:
        'https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/mmproj-F16.gguf',
  );

  static const gemma3_4b_q3_mm = LlmConfig(
    name: 'gemma-3-4b-it-Q3_K_M',
    url:
        'https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q3_K_M.gguf',
    imageUrl:
        'https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/mmproj-F16.gguf',
  );

  LlmConfig copyWith({
    String? name,
    String? url,
    String? imageUrl,
    ChatFormat? chatFormat,
    String? systemPrompt,
    int? contextSize,
    int? nPredict,
    int? nBatch,
    int? nThreads,
    double? temperature,
    int? topK,
    double? topP,
    double? minP,
    double? penaltyRepeat,
  }) => LlmConfig(
    name: name ?? this.name,
    url: url ?? this.url,
    imageUrl: imageUrl ?? this.imageUrl,
    chatFormat: chatFormat ?? this.chatFormat,
    systemPrompt: systemPrompt ?? this.systemPrompt,
    contextSize: contextSize ?? this.contextSize,
    nPredict: nPredict ?? this.nPredict,
    nBatch: nBatch ?? this.nBatch,
    nThreads: nThreads ?? this.nThreads,
    temperature: temperature ?? this.temperature,
    topK: topK ?? this.topK,
    topP: topP ?? this.topP,
    minP: minP ?? this.minP,
    penaltyRepeat: penaltyRepeat ?? this.penaltyRepeat,
  );
}
