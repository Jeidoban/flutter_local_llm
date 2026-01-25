import 'dart:ffi';
import 'dart:math';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';
// import 'package:ffi/ffi.dart';

/// Extended Llama class with automatic context trimming support
class LlamaWithTrim extends Llama {
  final bool _autoTrim;
  final ContextParams _contextParams;

  LlamaWithTrim(
    super.modelPath, {
    super.modelParams,
    super.contextParams,
    super.samplerParams,
    super.verbose,
    bool autoTrim = true,
  }) : _autoTrim = autoTrim,
       _contextParams = contextParams ?? ContextParams();

  /// Ensures there is room in the KV cache for [tokensNeeded] new tokens.
  (int, bool) _maybeTrimContext(int tokensNeeded) {
    if (!_autoTrim) return (0, false);

    final lib = getLib();
    final nCtx = _contextParams.nCtx;
    final mem = lib.llama_get_memory(context);

    final posMin = lib.llama_memory_seq_pos_min(mem, 0);
    final posMax = lib.llama_memory_seq_pos_max(mem, 0);

    if (posMin == -1 || posMax == -1) return (0, false);

    final currentLen = posMax - posMin + 1;
    final needed = tokensNeeded <= 0 ? 1 : tokensNeeded;

    final maxAllowed = nCtx - 5;
    if (currentLen + needed < maxAllowed) return (0, false);

    final overflow = (currentLen + needed) - (maxAllowed - 1);
    if (overflow <= 0) return (0, false);

    final removed = lib.llama_memory_seq_rm(mem, 0, posMin, posMin + overflow);
    if (!removed) return (0, false);

    if (!lib.llama_memory_can_shift(mem)) {
      lib.llama_memory_clear(mem, true);
      return (0, true);
    }

    lib.llama_memory_seq_add(
      mem,
      0,
      posMin + overflow,
      -1,
      -(posMin + overflow),
    );

    return (overflow, true);
  }

  @override
  (String, bool, bool) getNextWithStatus() {
    // Trim before decoding if needed
    if (batch.n_tokens > 0) {
      final (shifted, trimmed) = _maybeTrimContext(batch.n_tokens);

      if (trimmed && shifted > 0) {
        // Adjust batch positions to account for the shift
        for (int i = 0; i < batch.n_tokens; i++) {
          batch.pos[i] = max(0, batch.pos[i] - shifted);
        }
      }
    }

    // Call parent to decode and sample
    final result = super.getNextWithStatus();

    // After super sets up the next batch, correct the position based on actual KV cache state
    // This ensures positions stay correct even after trimming
    final lib = getLib();
    final mem = lib.llama_get_memory(context);
    final posMax = lib.llama_memory_seq_pos_max(mem, 0);

    if (posMax != -1 && batch.n_tokens > 0) {
      // Set the next token position to be right after the current max position
      batch.pos[0] = posMax + 1;
    }

    return result;
  }
}
