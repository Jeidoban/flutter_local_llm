import 'dart:io';
import 'package:flutter_local_llm/flutter_local_llm.dart';
import 'package:flutter_local_llm/src/llm_isolate.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

// Mock implementations using mocktail
class MockModelManager extends Mock implements ModelManager {}

class MockChatManager extends Mock implements ChatManager {}

class MockLLMIsolate extends Mock implements LlmIsolate {}

/// Register fallback values for mocktail
void registerMocktailFallbacks() {
  registerFallbackValue(LlmModel.gemma3_1b_q5);
  registerFallbackValue(LlmConfig.gemma3_1b_q5);
  registerFallbackValue(Role.user);
  registerFallbackValue(GenerateFromPromptCommand(prompt: '', requestId: 0));
  registerFallbackValue(ClearContextCommand());
  registerFallbackValue(GetRemainingContextCommand(requestId: 0));
}

/// Helper for managing temp directories in tests
class TempTestDirectory {
  late Directory dir;

  Future<void> setup() async {
    dir = await Directory.systemTemp.createTemp('flutter_local_llm_test_');
  }

  Future<void> teardown() async {
    if (dir.existsSync()) {
      await dir.delete(recursive: true);
    }
  }
}

/// Check if two timestamps are within tolerance
bool timestampsNear(
  DateTime a,
  DateTime b, {
  Duration tolerance = const Duration(seconds: 1),
}) {
  return a.difference(b).abs() < tolerance;
}
