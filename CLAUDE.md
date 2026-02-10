# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

flutter_local_llm is a Flutter plugin that enables running large language models (LLMs) locally on-device using llama.cpp. The plugin supports multimodal input (text and images) and provides both direct API access and integration with flutter_ai_toolkit.

## Architecture

### Core Components

**FlutterLocalLlm** (`lib/src/flutter_local_llm_base.dart`)
- Main entry point for using local LLMs
- Handles model downloading and caching to `flutter_local_llm_models/` directory
- Manages chat history and context trimming
- Provides streaming and complete response methods
- Key methods: `init()`, `sendMessage()`, `sendMessageWithHistory()`, `clearHistory()`

**LLMIsolate** (`lib/src/llm_isolate.dart`)
- Runs llama.cpp in a separate Dart isolate to prevent UI blocking
- Uses command-response pattern for communication between main and isolate threads
- Commands: `InitializeCommand`, `GenerateFromPromptCommand`, `ClearContextCommand`, `GetRemainingContextCommand`
- Responses: `TokenResponse`, `CompletionResponse`, `ErrorResponse`, `RemainingContextResponse`
- Platform-specific library loading (iOS/macOS frameworks, Android .so, Windows .dll)

**LocalLlmProvider** (`lib/src/local_llm_provider.dart`)
- Adapter for flutter_ai_toolkit's `LlmProvider` interface
- Wraps `FlutterLocalLlm` for use with `LlmChatView` widget
- Handles attachment file conversion (writes bytes to temporary files)
- Synchronizes flutter_ai_toolkit's ChatMessage history with internal LlmChatHistory

**LlmChatHistory** (`lib/src/llm_chat_history.dart`)
- Extends `ChatHistory` from llama_cpp_dart
- Implements automatic context window management
- Trims older messages when context space runs low, preserving system messages and recent pairs

### Native Integration

The plugin embeds llama.cpp as native frameworks via git submodule at `src/llama.cpp/`:
- **Build process**: Run `./build_llama.sh` to compile llama.cpp into frameworks
- **iOS**: Uses `llama.xcframework` (iOS and simulator only, macOS stripped)
- **macOS**: Uses `llama.framework` (macOS binary only)
- **Podspecs**: Reference vendored frameworks via `s.vendored_frameworks`

## Development Commands

### Build Native Frameworks
```bash
# Build llama.cpp and copy frameworks to ios/ and macos/
./build_llama.sh
```
This script:
1. Builds llama.xcframework from llama.cpp submodule
2. Extracts macOS framework to `macos/Frameworks/llama.framework`
3. Copies iOS xcframework to `ios/Frameworks/llama.xcframework` (without macOS slice)

### Run Example App
```bash
cd example
flutter run
```
The example demonstrates two usage patterns:
- `chat_screen.dart`: Direct FlutterLocalLlm usage with custom UI
- `ai_toolkit_chat_screen.dart`: LocalLlmProvider with LlmChatView widget

### Testing
```bash
flutter test
```

### Dependency Management
```bash
flutter pub get
```

## Key Implementation Details

### Message Flow
1. User sends message via `sendMessage()` or `sendMessageWithHistory()`
2. FlutterLocalLlm checks remaining context space
3. If context low, automatically trims history (keeps system messages + recent pairs)
4. Formats messages according to chat format (Gemma, ChatML, Alpaca)
5. Sends `GenerateFromPromptCommand` to isolate with prompt and optional attachment paths
6. Isolate streams back `TokenResponse` until completion
7. Response appended to chat history if `addToHistory: true`

### Context Management
- Context size configurable via `contextSize` parameter (default: 8096)
- Automatic trimming when `shouldTrimBeforePromptNoLlama()` returns true
- Preserves system messages and configurable number of recent message pairs
- Rough token estimation: ~4 characters per token

### Multimodal Support
- Models with `imageUrl` support multimodal input (e.g., gemma3_4b_q5_mm)
- Downloads both text model (`.gguf`) and image projection model (`mmproj-F16.gguf`)
- Image attachments passed as `List<File>` to `sendMessage()`
- Isolate converts files to `LlamaImage` and uses `generateWithMedia()`

### Supported Models
Models defined in `lib/src/models.dart`:
- `gemma3n_E2B_q4`: Text-only, 2B parameters, Q4 quantization
- `gemma3_4b_q5_mm`: Multimodal, 4B parameters, Q5 quantization
- `gemma3_4b_q3_mm`: Multimodal, 4B parameters, Q3 quantization
- `gemma3_1b_q5`: Text-only, 1B parameters, Q5 quantization

Custom models supported via `customModelUrl` and `customImageModelUrl` parameters.

## Platform-Specific Notes

### iOS/macOS
- Frameworks must be built before first run using `./build_llama.sh`
- Metal acceleration used automatically (via ggml-metal)
- iOS requires device or simulator with arm64 architecture

### Android/Linux/Windows
- Uses shared library (.so/.dll) from llama_cpp_dart package
- Library path configured in `LlamaManager._setupLlamaLibraryPath()`

## Common Patterns

### Basic Usage
```dart
final llm = await FlutterLocalLlm.init(
  model: LLMModel.gemma3_4b_q5_mm,
  systemPrompt: 'You are a helpful assistant.',
);

await for (final token in llm.sendMessage('Hello!')) {
  print(token); // Stream tokens
}

llm.dispose();
```

### With flutter_ai_toolkit
```dart
final llm = await FlutterLocalLlm.init(model: LLMModel.gemma3_4b_q5_mm);
final provider = LocalLlmProvider(llm: llm);

LlmChatView(provider: provider); // Use pre-built chat UI
```

### Multimodal Input
```dart
await for (final token in llm.sendMessage(
  'Describe this image',
  images: [File('/path/to/image.jpg')],
)) {
  print(token);
}
```

## Branch Context
Currently on branch: `add-multiple-chats`
Main branch: `main`
