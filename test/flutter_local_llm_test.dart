import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_local_llm/flutter_local_llm.dart';

void main() {
  test('FlutterLocalLlm exports are available', () {
    expect(LlmModel.values, isNotEmpty);
    expect(Role.values, isNotEmpty);
    expect(ChatFormat.values, isNotEmpty);
  });

  test('LlmModel enum has all models', () {
    expect(LlmModel.values, contains(LlmModel.gemma3n_E2B_q4));
    expect(LlmModel.values, contains(LlmModel.gemma3_4b_q5_mm));
    expect(LlmModel.values, contains(LlmModel.gemma3_4b_q3_mm));
    expect(LlmModel.values, contains(LlmModel.gemma3_1b_q5));
  });

  test('Role enum is available', () {
    expect(Role.user, isNotNull);
    expect(Role.assistant, isNotNull);
    expect(Role.system, isNotNull);
  });

  test('LlmConfig presets have sensible defaults', () {
    final config = LlmConfig.gemma3_4b_q5_mm;
    expect(config.contextSize, 16384);
    expect(config.temperature, 0.7);
    expect(config.topK, 64);
    expect(config.topP, 0.95);
    expect(config.penaltyRepeat, 1.1);
  });
}
