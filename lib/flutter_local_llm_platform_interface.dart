import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'flutter_local_llm_method_channel.dart';

abstract class FlutterLocalLlmPlatform extends PlatformInterface {
  FlutterLocalLlmPlatform() : super(token: _token);

  static final Object _token = Object();

  static FlutterLocalLlmPlatform _instance = MethodChannelFlutterLocalLlm();

  static FlutterLocalLlmPlatform get instance => _instance;

  static set instance(FlutterLocalLlmPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }
}
