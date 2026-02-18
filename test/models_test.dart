import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_local_llm/src/models.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() {
  group('LLMConfig', () {
    test('uses custom URLs over model defaults', () {
      final config = LLMConfig(
        model: LLMModel.gemma3_1b_q5,
        customModelUrl: 'https://custom.com/model.gguf',
        customImageModelUrl: 'https://custom.com/mmproj.gguf',
      );

      expect(config.downloadUrl, 'https://custom.com/model.gguf');
      expect(config.fileName, 'model.gguf');
      expect(config.imageDownloadUrl, 'https://custom.com/mmproj.gguf');
      expect(config.imageFileName, 'mmproj.gguf');
    });

    test('falls back to model URLs when no custom URLs', () {
      final config = LLMConfig(model: LLMModel.gemma3_1b_q5);

      expect(config.downloadUrl, LLMModel.gemma3_1b_q5.url);
      expect(config.fileName, LLMModel.gemma3_1b_q5.fileName);
      expect(config.imageDownloadUrl, null);
      expect(config.imageFileName, null);
    });

    test('chatFormat inherits from model or uses override', () {
      // Without explicit chatFormat → inherits from model
      final defaultConfig = LLMConfig(model: LLMModel.gemma3_1b_q5);
      expect(defaultConfig.chatFormat, LLMModel.gemma3_1b_q5.chatFormat);

      // With explicit chatFormat → overrides model
      final overrideConfig = LLMConfig(
        model: LLMModel.gemma3_1b_q5,
        chatFormat: ChatFormat.chatml,
      );
      expect(overrideConfig.chatFormat, ChatFormat.chatml);
    });
  });

  group('LLMModel', () {
    test('all LLMModel enums have complete properties', () {
      for (final model in LLMModel.values) {
        expect(model.name, isNotEmpty);
        expect(model.url, startsWith('https://'));
        expect(model.fileName, endsWith('.gguf'));
        expect(model.chatFormat, ChatFormat.gemma); // All current models are Gemma
      }
    });

    test('multimodal models have image URLs, text-only do not', () {
      // Multimodal models
      expect(LLMModel.gemma3_4b_q5_mm.imageUrl, isNotNull);
      expect(LLMModel.gemma3_4b_q5_mm.imageFileName, contains('mmproj'));
      expect(LLMModel.gemma3_4b_q3_mm.imageUrl, isNotNull);
      expect(LLMModel.gemma3_4b_q3_mm.imageFileName, contains('mmproj'));

      // Text-only models
      expect(LLMModel.gemma3n_E2B_q4.imageUrl, isNull);
      expect(LLMModel.gemma3n_E2B_q4.imageFileName, isNull);
      expect(LLMModel.gemma3_1b_q5.imageUrl, isNull);
      expect(LLMModel.gemma3_1b_q5.imageFileName, isNull);
    });
  });
}
