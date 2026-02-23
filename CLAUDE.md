# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

flutter_local_llm is a Flutter plugin that runs large language models locally on-device via llama.cpp. It supports multimodal input (text + images) and integrates with flutter_ai_toolkit.

## Architecture

The library is designed around dependency injection. `FlutterLocalLlm` is the main entry point and depends on two injectable classes: `ModelManager` and `ChatManager`. This makes all external concerns (file I/O, networking, isolate creation) mockable in tests.

### Core Components

**`FlutterLocalLlm`** (`lib/src/flutter_local_llm_base.dart`)

The main API surface. Orchestrates model loading, message generation, and chat persistence. Has two factory constructors: a simple one with sensible defaults (`create`), and one that accepts explicit dependencies for testing (`createCustom`). Read this file for the current public API.

**`ModelManager`** (`lib/src/model_manager.dart`)

Handles downloading model files from the network and creating the LLM isolate. Has a configurable models directory (defaults to the app documents folder). Can be mocked in tests to avoid real downloads and isolate creation.

**`ChatManager`** (`lib/src/chat_manager.dart`)

Owns all chat state: the list of chat histories, the active chat index, and persistence to disk. Has a configurable storage path (defaults to the app documents folder). Can be mocked in tests to avoid file I/O.

**`LlmIsolate`** (`lib/src/llm_isolate.dart`)

Runs llama.cpp in a separate Dart isolate to keep the UI responsive. Uses a command/response pattern: the main thread sends typed command objects and receives typed response objects over `SendPort`/`ReceivePort`. Commands and responses are defined in the same file. Not directly testable without the native library — test the command/response data structures and `LlamaManager`'s error handling separately.

**`LlmChatHistory`** (`lib/src/llm_chat_history.dart`)

Extends `ChatHistory` from llama_cpp_dart with metadata (title, timestamps) and persistence support. Maintains two views of history: a complete archive (`fullHistory`) and a sliding context window (`messages`) that fits within the model's context limit. Image token cost is estimated to determine when to trim.

**`LocalLlmProvider`** (`lib/src/local_llm_provider.dart`)

Adapter for flutter_ai_toolkit's `LlmProvider`. Wraps `FlutterLocalLlm` and translates between the toolkit's `ChatMessage` format and the internal message format. Handles writing attachment bytes to temp files for multimodal input.

**`LlmModel` / `LlmConfig`** (`lib/src/models.dart`)

`LlmModel` is an enum of supported models. `LlmConfig` is the runtime configuration (context size, sampling params, URLs, file names). Each `LlmModel` value has a corresponding `LlmConfig` preset. Custom models are supported by constructing `LlmConfig` directly.

### Native Integration

llama.cpp is embedded as a native framework via a git submodule at `src/llama.cpp/`. Platform details:
- **iOS/macOS**: Vendored frameworks built from source via `./build_llama.sh`. Metal acceleration is automatic.
- **Android/Linux/Windows**: Uses shared libraries from the `llama_cpp_dart` package.

## Development Commands

```bash
# Build native frameworks (iOS/macOS only, required before first run)
./build_llama_apple.sh

# Run tests
flutter test

# Run the example app
cd example && flutter run

# Fetch dependencies
flutter pub get
```

## Key Design Decisions

**Dependency injection for testability**: `ModelManager` and `ChatManager` are concrete classes (not interfaces) that can be mocked with mocktail. The `createCustom` factory accepts them as parameters. This lets tests run without network access, file system access, or real isolates.

**Chat management is separated from the LLM**: `ChatManager` owns all chat state and persistence independently of the isolate. Switching chats clears the isolate's context to prevent history from bleeding between sessions.

**Context window is managed automatically**: `LlmChatHistory` tracks how much context has been consumed and trims older messages when space runs low. System messages are always preserved. The number of recent message pairs to keep is configurable.

**Full history vs. active context**: Every message is stored in `fullHistory`. Only the subset that fits in the context window is in `messages`. When persisting to disk, `fullHistory` is saved so no conversation data is lost.

**Auto-titling and auto-creation**: A new chat is created automatically if none exists when the first message is sent. The chat title is set from the content of the first user message.

**Multimodal**: Models that support images download a second projection model file alongside the main model. Images are passed as file paths to the isolate, which handles conversion to the format llama.cpp expects.

## File Storage

All data lives under the app's documents directory by default:

- **Models**: `flutter_local_llm/models/` — `.gguf` model files
- **Chat data**: `flutter_local_llm/data/chats.json` — all chat histories, pretty-printed JSON

Both paths are configurable when constructing `ModelManager` and `ChatManager`.

## Testing

Tests live in `test/`. The suite uses `mocktail` for mocking. Key patterns:

- `ModelManager` and `ChatManager` are mocked to avoid file I/O and network calls
- `FlutterLocalLlm` is created via `createCustom` with mock dependencies
- Isolate responses are simulated with a `StreamController<IsolateResponse>` on a broadcast stream — responses must be emitted asynchronously (e.g. via `Future.microtask`) to avoid race conditions with `await for` listeners
- `LlamaManager` (inside `llm_isolate.dart`) has a public constructor that accepts a `SendPort`, making its error-handling testable without native code
- When mocking `sendMessage` on `FlutterLocalLlm`, use `any(named: ...)` for all named parameters — mocktail captures default argument values at stub registration time, so calls with explicit values won't match stubs registered with defaults

See `test/test_helpers.dart` for shared mock classes and fallback value registration.
