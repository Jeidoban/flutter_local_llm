import 'llm_isolate.dart';
import 'models.dart';

/// Manages LLM isolate lifecycle
///
/// Wraps the static LLMIsolate.spawn() method to make it mockable for testing
class IsolateManager {
  /// Create and initialize an isolate with the model
  Future<LLMIsolate> createIsolate(
    String modelPath,
    LLMConfig config, {
    String? imageModelPath,
  }) async {
    return await LLMIsolate.spawn(
      modelPath,
      config,
      imageModelPath: imageModelPath,
    );
  }
}
