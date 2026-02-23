import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'flutter_local_llm_platform_interface.dart';

/// An implementation of [FlutterLocalLlmPlatform] that uses method channels.
class MethodChannelFlutterLocalLlm extends FlutterLocalLlmPlatform {
  @visibleForTesting
  final methodChannel = const MethodChannel('flutter_local_llm');
}
