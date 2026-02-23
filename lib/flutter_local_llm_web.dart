import 'package:flutter_web_plugins/flutter_web_plugins.dart';

import 'flutter_local_llm_platform_interface.dart';

/// A web implementation of the FlutterLocalLlmPlatform of the FlutterLocalLlm plugin.
class FlutterLocalLlmWeb extends FlutterLocalLlmPlatform {
  FlutterLocalLlmWeb();

  static void registerWith(Registrar registrar) {
    FlutterLocalLlmPlatform.instance = FlutterLocalLlmWeb();
  }
}
