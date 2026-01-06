import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'flutter_local_llm_method_channel.dart';

abstract class FlutterLocalLlmPlatform extends PlatformInterface {
  /// Constructs a FlutterLocalLlmPlatform.
  FlutterLocalLlmPlatform() : super(token: _token);

  static final Object _token = Object();

  static FlutterLocalLlmPlatform _instance = MethodChannelFlutterLocalLlm();

  /// The default instance of [FlutterLocalLlmPlatform] to use.
  ///
  /// Defaults to [MethodChannelFlutterLocalLlm].
  static FlutterLocalLlmPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [FlutterLocalLlmPlatform] when
  /// they register themselves.
  static set instance(FlutterLocalLlmPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }
}
