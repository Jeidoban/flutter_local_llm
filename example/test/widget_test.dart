import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_local_llm_example/main.dart';

void main() {
  testWidgets('HomeScreen renders navigation bar', (WidgetTester tester) async {
    await tester.pumpWidget(const MyApp());

    expect(find.text('Manual'), findsOneWidget);
    expect(find.text('Toolkit'), findsOneWidget);
    expect(find.text('Settings'), findsOneWidget);
  });
}
