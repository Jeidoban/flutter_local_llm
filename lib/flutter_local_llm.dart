// Public API exports
export 'src/flutter_local_llm_base.dart';
export 'src/local_llm_provider.dart';
export 'src/models.dart';
export 'src/llm_chat_history.dart';

// Dependency injection classes (can be mocked in tests)
export 'src/model_manager.dart';
export 'src/chat_storage.dart';
export 'src/isolate_manager.dart';

// Re-export useful types from llama_cpp_dart that users might need
export 'package:llama_cpp_dart/llama_cpp_dart.dart'
    show ChatFormat, Message, Role;
