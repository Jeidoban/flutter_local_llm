import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

/// Set up the llama library path based on the current platform
void setupLlamaLibraryPath() {
  // For iOS/macOS, the framework is embedded and linked via CocoaPods
  // FFI needs the library name that dyld can find via @rpath

  if (Platform.isMacOS) {
    // On macOS, use the framework path - dyld will find it via @rpath
    // since it's embedded in the app bundle via CocoaPods
    Llama.libraryPath = 'llama.framework/llama';
    if (kDebugMode) {
      print('[setupLlamaLibraryPath] Set library path for macOS');
    }
  } else if (Platform.isIOS) {
    // On iOS, the xcframework is embedded via CocoaPods
    // The FFI will find it automatically via @rpath
    // No need to set the path explicitly for iOS
    if (kDebugMode) {
      print('[setupLlamaLibraryPath] Using default library path for iOS');
    }
  } else if (Platform.isAndroid) {
    // For Android, set the path to the .so file
    Llama.libraryPath = 'libllama.so';
    if (kDebugMode) {
      print('[setupLlamaLibraryPath] Set library path for Android');
    }
  } else if (Platform.isLinux) {
    Llama.libraryPath = 'libllama.so';
    if (kDebugMode) {
      print('[setupLlamaLibraryPath] Set library path for Linux');
    }
  } else if (Platform.isWindows) {
    Llama.libraryPath = 'llama.dll';
    if (kDebugMode) {
      print('[setupLlamaLibraryPath] Set library path for Windows');
    }
  }
}
