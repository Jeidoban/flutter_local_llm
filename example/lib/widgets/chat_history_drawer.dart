import 'package:flutter/material.dart';
import 'package:flutter_local_llm/flutter_local_llm.dart';

class ChatHistoryDrawer extends StatelessWidget {
  final List<LlmChatHistory> chats;
  final int? activeIndex;
  final void Function(int) onSelectChat;
  final VoidCallback onNewChat;
  final void Function(int) onDeleteChat;

  const ChatHistoryDrawer({
    super.key,
    required this.chats,
    required this.activeIndex,
    required this.onSelectChat,
    required this.onNewChat,
    required this.onDeleteChat,
  });

  String _formatDate(DateTime dt) {
    final now = DateTime.now();
    final diff = now.difference(dt);
    if (diff.inDays == 0) return 'Today';
    if (diff.inDays == 1) return 'Yesterday';
    return '${dt.month}/${dt.day}/${dt.year}';
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Drawer(
      child: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
              child: Text(
                'Conversations',
                style: theme.textTheme.titleMedium,
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              child: FilledButton.icon(
                icon: const Icon(Icons.add),
                label: const Text('New Chat'),
                onPressed: () {
                  onNewChat();
                  Navigator.pop(context);
                },
              ),
            ),
            const Divider(),
            Expanded(
              child: chats.isEmpty
                  ? Center(
                      child: Text(
                        'No conversations yet',
                        style: theme.textTheme.bodyMedium?.copyWith(
                          color: theme.colorScheme.outline,
                        ),
                      ),
                    )
                  : ListView.builder(
                      padding: EdgeInsets.zero,
                      itemCount: chats.length,
                      itemBuilder: (context, index) {
                        final chat = chats[index];
                        final isActive = index == activeIndex;
                        return ListTile(
                          selected: isActive,
                          selectedTileColor: theme.colorScheme.primaryContainer
                              .withValues(alpha: 0.3),
                          title: Text(
                            chat.title,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                          subtitle: Text(_formatDate(chat.updatedAt)),
                          trailing: IconButton(
                            icon: const Icon(Icons.delete_outline, size: 20),
                            onPressed: () => onDeleteChat(index),
                          ),
                          onTap: () {
                            onSelectChat(index);
                            Navigator.pop(context);
                          },
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
