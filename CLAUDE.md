# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

flutter_local_llm is a Flutter plugin that enables running large language models (LLMs) locally on-device using llama.cpp. The plugin supports multimodal input (text and images) and provides both direct API access and integration with flutter_ai_toolkit.

## Architecture

### Core Components

**FlutterLocalLlm** (`lib/src/flutter_local_llm_base.dart`)
- Main entry point for using local LLMs
- Handles model downloading and caching to `flutter_local_llm/models/` directory
- Manages multiple chat sessions with automatic persistence to `flutter_local_llm/data/chats.json`
- Provides streaming and complete response methods
- Chat management: `startNewChat()`, `setActiveChat()`, `deleteChat()`, `deleteAllChats()`, `saveChats()`
- Message methods: `init()`, `sendMessage()`, `sendMessageWithHistory()`, `clearHistory()`
- Auto-creates chat on first message if none exists
- Clears LLM context when switching between chats to prevent cross-contamination

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
- Extends `ChatHistory` from llama_cpp_dart with metadata fields: `title`, `createdAt`, `updatedAt`
- Maintains both `fullHistory` (complete conversation) and `messages` (active context window)
- Implements automatic context window management with image token estimation (300 tokens per image)
- Trims older messages when context space runs low, preserving system messages and recent pairs
- Custom `toJson()`/`fromJson()` methods for persistence that serialize `fullHistory` and restore active context
- Tracks `_contextStartIndex` to know which portion of fullHistory is in active context

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
2. FlutterLocalLlm auto-creates a new chat if none exists
3. Checks remaining context space from isolate
4. If context low or empty, rebuilds/trims history (keeps system messages + recent pairs)
5. Formats messages according to chat format (Gemma, ChatML, Alpaca)
6. Sends `GenerateFromPromptCommand` to isolate with prompt and optional attachment paths
7. Isolate streams back `TokenResponse` until completion
8. If `addToHistory: true`:
   - Auto-titles chat from first user message (if still "New Chat")
   - Adds messages and response to active chat's history
   - Updates `updatedAt` timestamp
   - Saves all chats to `chats.json`

### Context Management
- Context size configurable via `contextSize` parameter (default: 8096)
- Automatic trimming when `shouldTrimBeforePromptNoLlama()` returns true
  - Triggers at 80% capacity (4/5 of context) to leave room for long responses
  - Includes estimation for both text (~4 chars per token) and images (300 tokens each)
- Preserves system messages and configurable number of recent message pairs
- `keepRecentPairs` calculated automatically based on context size (contextSize / 2048, clamped 1-10)
- Context cleared when switching between chats to prevent history bleeding

### Multi-Chat Management
- Supports multiple independent chat sessions stored in a list
- Active chat accessed via `activeChat` getter (auto-creates if none exists)
- Each chat has metadata: `title`, `createdAt`, `updatedAt` timestamps
- Auto-titling from first user message (first 40 characters, or full message if shorter)
- Persistence to `flutter_local_llm/data/chats.json` with pretty-printed JSON
- Saves automatically after every message exchange with `updatedAt` timestamp update
- Chat operations:
  - `startNewChat({String? title})`: Create new chat with optional title (default: "New Chat")
  - `setActiveChat(int index)`: Switch to different chat by index, clears LLM context
  - `deleteChat(int index)`: Remove chat, adjusts active index if needed
  - `deleteAllChats()`: Clear all chats and delete storage file
  - `saveChats()`: Manually save all chats (updates all `updatedAt` timestamps)
- Access chat list via `chats` getter, active chat index via `activeChatIndex`
- List-based storage: chat index serves as identifier (no separate ID field)

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
final llm = await FlutterLocalLlm.create(
  model: LLMModel.gemma3_1b_q5,  // Default model
  systemPrompt: 'You are a helpful assistant.',
);

await for (final token in llm.sendMessage('Hello!')) {
  print(token); // Stream tokens
}

llm.dispose();
```

### Multi-Chat Management
```dart
// Initialize with automatic chat creation
final llm = await FlutterLocalLlm.create();

// First message auto-creates a chat and sets its title
await for (final token in llm.sendMessage('What is Flutter?')) {
  print(token);
}

// Create additional chats
final chatIndex1 = llm.startNewChat(title: 'Flutter Questions');
final chatIndex2 = llm.startNewChat(title: 'Dart Questions');

// Switch between chats
llm.setActiveChat(chatIndex1);

// Access all chats
for (var i = 0; i < llm.chats.length; i++) {
  print('${llm.chats[i].title} - ${llm.chats[i].messages.length} messages');
}

// Delete a chat
await llm.deleteChat(chatIndex2);

// Clear all chats
await llm.deleteAllChats();

// Manually save after modifying chat properties
llm.chats[0].title = 'Updated Title';
await llm.saveChats();
```

### With flutter_ai_toolkit
```dart
final llm = await FlutterLocalLlm.create(model: LLMModel.gemma3_4b_q5_mm);
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

## File Storage Locations

All data is stored in the application's documents directory:

- **Models**: `<app_docs>/flutter_local_llm/models/*.gguf`
  - Text model files (e.g., `gemma-3-1b-it-Q5_K_M.gguf`)
  - Image projection models (e.g., `gemma-3-4b-it-Q5_K_M-mmproj-F16.gguf`)

- **Chat Data**: `<app_docs>/flutter_local_llm/data/chats.json`
  - All chat histories with metadata
  - Pretty-printed JSON for debugging
  - Contains: `activeChatIndex` and array of chat objects
  - Each chat includes: `title`, `createdAt`, `updatedAt`, `messages` (fullHistory), `contextStartIndex`

**Note**: The plugin automatically migrates from the old `flutter_local_llm_models/` directory structure to the new `flutter_local_llm/models/` and `flutter_local_llm/data/` structure.

## Testing

The library is fully testable using dependency injection. All external dependencies (model downloading, file storage, isolate creation) can be mocked for testing.

### Testing with Mocktail

```dart
import 'package:flutter_local_llm/flutter_local_llm.dart';
import 'package:mocktail/mocktail.dart';

// Create mocks
class MockModelManager extends Mock implements ModelManager {}
class MockChatStorage extends Mock implements ChatStorage {}
class MockIsolateManager extends Mock implements IsolateManager {}

// In your test
void main() {
  test('example test', () async {
    final mockModelManager = MockModelManager();
    final mockChatStorage = MockChatStorage();
    final mockIsolateManager = MockIsolateManager();

    // Set up mocks
    when(() => mockModelManager.getModelPath(any(), any(), any()))
        .thenAnswer((_) async => '/fake/model.gguf');
    when(() => mockIsolateManager.createIsolate(any(), any(), imageModelPath: any(named: 'imageModelPath')))
        .thenAnswer((_) async => mockIsolateHandle);
    when(() => mockChatStorage.loadChats())
        .thenAnswer((_) async => null);

    // Create LLM with mocked dependencies
    final llm = await FlutterLocalLlm.createWithDependencies(
      modelManager: mockModelManager,
      chatStorage: mockChatStorage,
      isolateManager: mockIsolateManager,
    );

    // Test your logic...
  });
}
```

### Custom Implementations

You can provide custom implementations for enterprise use cases:

```dart
// Custom model manager that downloads from cloud storage
class CloudModelManager implements ModelManager {
  @override
  Future<String> getModelPath(String url, String fileName, Function(double)? onProgress) async {
    // Download from your cloud storage
  }

  @override
  Future<Directory> getModelsDirectory() async {
    // Return custom directory
  }
}

// Use custom implementation
final llm = await FlutterLocalLlm.createWithDependencies(
  modelManager: CloudModelManager(),
  chatStorage: ChatStorage(),
  isolateManager: IsolateManager(),
);
```
