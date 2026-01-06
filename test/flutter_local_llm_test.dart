import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_local_llm/flutter_local_llm.dart';
import 'package:flutter_local_llm/flutter_local_llm_platform_interface.dart';
import 'package:flutter_local_llm/flutter_local_llm_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockFlutterLocalLlmPlatform
    with MockPlatformInterfaceMixin
    implements FlutterLocalLlmPlatform {

  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  final FlutterLocalLlmPlatform initialPlatform = FlutterLocalLlmPlatform.instance;

  test('$MethodChannelFlutterLocalLlm is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelFlutterLocalLlm>());
  });

  test('getPlatformVersion', () async {
    FlutterLocalLlm flutterLocalLlmPlugin = FlutterLocalLlm();
    MockFlutterLocalLlmPlatform fakePlatform = MockFlutterLocalLlmPlatform();
    FlutterLocalLlmPlatform.instance = fakePlatform;

    expect(await flutterLocalLlmPlugin.getPlatformVersion(), '42');
  });
}
