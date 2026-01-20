import 'package:flutter/material.dart';
import 'chat_screen.dart';
import 'ai_toolkit_chat_screen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Local LLM',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Flutter Local LLM Examples'),
        elevation: 2,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Padding(
              padding: EdgeInsets.all(24),
              child: Text(
                'Choose an example to explore:',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w500),
              ),
            ),
            const SizedBox(height: 16),
            _ExampleButton(
              icon: Icons.chat_bubble_outline,
              label: 'Custom Chat UI',
              description: 'Direct FlutterLocalLlm usage with custom UI',
              onPressed: () => Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const ChatScreen()),
              ),
            ),
            const SizedBox(height: 16),
            _ExampleButton(
              icon: Icons.auto_awesome,
              label: 'AI Toolkit Chat',
              description: 'Uses LocalLlmProvider with LlmChatView',
              onPressed: () => Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const AiToolkitChatScreen()),
              ),
            ),
            const SizedBox(height: 48),
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 32),
              child: Text(
                'Both examples use the same underlying local LLM engine.\n'
                'The first shows custom UI implementation,\n'
                'while the second uses flutter_ai_toolkit\'s pre-built components.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 12, color: Colors.grey),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ExampleButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final String description;
  final VoidCallback onPressed;

  const _ExampleButton({
    required this.icon,
    required this.label,
    required this.description,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 280,
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          padding: const EdgeInsets.all(20),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 32),
            const SizedBox(height: 8),
            Text(
              label,
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 4),
            Text(
              description,
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 12,
                color: Theme.of(context).colorScheme.onSurface.withValues(alpha: 0.7),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
