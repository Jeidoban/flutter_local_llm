import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_local_llm/src/models/llm_config.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() {
  group('LlmConfig presets', () {
    test('all presets have required fields', () {
      for (final config in [
        LlmConfig.gemma3_1b_q5,
        LlmConfig.gemma3n_E2B_q4,
        LlmConfig.gemma3_4b_q5_mm,
        LlmConfig.gemma3_4b_q3_mm,
      ]) {
        expect(config.name, isNotEmpty);
        expect(config.url, startsWith('https://'));
        expect(config.fileName, endsWith('.gguf'));
        expect(config.chatFormat, ChatFormat.gemma);
      }
    });

    test('multimodal presets have imageUrl, text-only do not', () {
      expect(LlmConfig.gemma3_4b_q5_mm.imageUrl, isNotNull);
      expect(LlmConfig.gemma3_4b_q3_mm.imageUrl, isNotNull);
      expect(LlmConfig.gemma3_1b_q5.imageUrl, isNull);
      expect(LlmConfig.gemma3n_E2B_q4.imageUrl, isNull);
    });

    test('imageFileName is derived from name and imageUrl', () {
      expect(LlmConfig.gemma3_4b_q5_mm.imageFileName, contains('mmproj'));
      expect(LlmConfig.gemma3_4b_q3_mm.imageFileName, contains('mmproj'));
      expect(LlmConfig.gemma3_1b_q5.imageFileName, isNull);
    });

    test('fileName is name + .gguf', () {
      expect(LlmConfig.gemma3_1b_q5.fileName, 'gemma-3-1b-it-Q5_K_M.gguf');
    });
  });

  group('LlmConfig copyWith', () {
    test('overrides specified fields', () {
      const custom = 'https://example.com/my-model.gguf';
      final config = LlmConfig.gemma3_1b_q5.copyWith(
        url: custom,
        contextSize: 4096,
        systemPrompt: 'Custom prompt',
      );

      expect(config.url, custom);
      expect(config.contextSize, 4096);
      expect(config.systemPrompt, 'Custom prompt');
      // Unchanged fields preserved
      expect(config.name, LlmConfig.gemma3_1b_q5.name);
      expect(config.chatFormat, LlmConfig.gemma3_1b_q5.chatFormat);
    });

    test('preserves all fields when nothing overridden', () {
      final copy = LlmConfig.gemma3_4b_q5_mm.copyWith();
      expect(copy.name, LlmConfig.gemma3_4b_q5_mm.name);
      expect(copy.url, LlmConfig.gemma3_4b_q5_mm.url);
      expect(copy.imageUrl, LlmConfig.gemma3_4b_q5_mm.imageUrl);
      expect(copy.contextSize, LlmConfig.gemma3_4b_q5_mm.contextSize);
    });
  });

  group('LlmModel enum', () {
    test('has all four models', () {
      expect(LlmModel.values.length, 4);
      expect(LlmModel.values, contains(LlmModel.gemma3_1b_q5));
      expect(LlmModel.values, contains(LlmModel.gemma3n_E2B_q4));
      expect(LlmModel.values, contains(LlmModel.gemma3_4b_q5_mm));
      expect(LlmModel.values, contains(LlmModel.gemma3_4b_q3_mm));
    });
  });
}
