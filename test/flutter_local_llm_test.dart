import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_local_llm/flutter_local_llm.dart';

void main() {
  test('FlutterLocalLlm exports are available', () {
    // Verify that the public types are exported correctly
    expect(LLMModel.values, isNotEmpty);
    expect(Role.values, isNotEmpty);
    expect(ChatFormat.values, isNotEmpty);
  });

  test('LLMModel has correct properties', () {
    final model = LLMModel.gemma3n_E2BTextOnly;
    expect(model.name, isNotEmpty);
    expect(model.url, isNotEmpty);
    expect(model.fileName, isNotEmpty);
    expect(model.chatFormat, ChatFormat.gemma);
  });

  test('Role enum is available', () {
    expect(Role.user, isNotNull);
    expect(Role.assistant, isNotNull);
    expect(Role.system, isNotNull);
  });

  test('LLMConfig has sensible defaults', () {
    final config = LLMConfig();
    expect(config.model, LLMModel.gemma3n_E2BTextOnly);
    expect(config.contextSize, 8192);
    expect(config.temperature, 0.7);
    expect(config.topK, 64);
    expect(config.topP, 0.95);
    expect(config.penaltyRepeat, 1.1);
  });
}
