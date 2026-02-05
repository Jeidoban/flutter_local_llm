// This is a basic Flutter integration test.
//
// Since integration tests run in a full Flutter application, they can interact
// with the host side of a plugin implementation, unlike Dart unit tests.
//
// For more information about Flutter integration tests, please see
// https://flutter.dev/to/integration-testing

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'package:flutter_local_llm/flutter_local_llm.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('FlutterLocalLlm API is available', (WidgetTester tester) async {
    // Verify that the API types are accessible
    expect(LLMModel.gemma3n_E2B_q4, isNotNull);
    expect(Role.user, isNotNull);

    // Note: We don't actually initialize the model here as it would be slow
    // and require downloading a large file. This test just verifies the API
    // is available and properly exported.
  });
}
