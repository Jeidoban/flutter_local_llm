import 'package:flutter/material.dart';
import 'package:flutter_local_llm/flutter_local_llm.dart';

class ModelPicker extends StatelessWidget {
  final LlmModel current;
  final bool enabled;
  final void Function(LlmModel) onSelected;

  const ModelPicker({
    super.key,
    required this.current,
    required this.enabled,
    required this.onSelected,
  });

  static String labelFor(LlmModel model) => switch (model) {
        LlmModel.gemma3_1b_q5 => 'Gemma 3 1B',
        LlmModel.gemma3n_E2B_q4 => 'Gemma 3n E2B',
        LlmModel.gemma3_4b_q5_mm => 'Gemma 3 4B Q5 (Multimodal)',
        LlmModel.gemma3_4b_q3_mm => 'Gemma 3 4B Q3 (Multimodal)',
      };

  @override
  Widget build(BuildContext context) {
    return PopupMenuButton<LlmModel>(
      icon: const Icon(Icons.model_training),
      tooltip: 'Switch model',
      enabled: enabled,
      onSelected: onSelected,
      itemBuilder: (_) => LlmModel.values
          .map(
            (model) => PopupMenuItem<LlmModel>(
              value: model,
              child: Row(
                children: [
                  SizedBox(
                    width: 24,
                    child: model == current
                        ? const Icon(Icons.check, size: 18)
                        : null,
                  ),
                  const SizedBox(width: 8),
                  Text(labelFor(model)),
                ],
              ),
            ),
          )
          .toList(),
    );
  }
}
