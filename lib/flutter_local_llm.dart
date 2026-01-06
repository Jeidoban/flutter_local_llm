
import 'flutter_local_llm_platform_interface.dart';

class FlutterLocalLlm {
  Future<String?> getPlatformVersion() {
    return FlutterLocalLlmPlatform.instance.getPlatformVersion();
  }
}
