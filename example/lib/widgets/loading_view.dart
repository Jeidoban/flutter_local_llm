import 'package:flutter/material.dart';

class LoadingView extends StatelessWidget {
  final String status;
  final double? progress;

  const LoadingView({super.key, required this.status, this.progress});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const CircularProgressIndicator(),
            const SizedBox(height: 24),
            Text(status, style: theme.textTheme.bodyLarge),
            if (progress != null) ...[
              const SizedBox(height: 16),
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(value: progress, minHeight: 8),
              ),
              const SizedBox(height: 8),
              Text(
                '${(progress! * 100).toStringAsFixed(0)}%',
                style: theme.textTheme.bodySmall,
              ),
            ],
          ],
        ),
      ),
    );
  }
}
