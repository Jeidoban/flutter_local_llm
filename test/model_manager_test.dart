import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_local_llm/src/managers/model_manager.dart';
import 'test_helpers.dart';

void main() {
  group('ModelManager.listModels', () {
    late TempTestDirectory tmp;
    late ModelManager manager;

    setUp(() async {
      tmp = TempTestDirectory();
      await tmp.setup();
      manager = ModelManager(modelsPath: tmp.dir.path);
    });

    tearDown(() => tmp.teardown());

    test('returns empty list when directory is empty', () async {
      final models = await manager.listModels();
      expect(models, isEmpty);
    });

    test('returns file names for existing model files', () async {
      File('${tmp.dir.path}/model-a.gguf').writeAsBytesSync([]);
      File('${tmp.dir.path}/model-b.gguf').writeAsBytesSync([]);

      final models = await manager.listModels();

      expect(models, unorderedEquals(['model-a.gguf', 'model-b.gguf']));
    });

    test('does not include subdirectories', () async {
      File('${tmp.dir.path}/model.gguf').writeAsBytesSync([]);
      Directory('${tmp.dir.path}/subdir').createSync();

      final models = await manager.listModels();

      expect(models, equals(['model.gguf']));
    });
  });

  group('ModelManager.deleteModel', () {
    late TempTestDirectory tmp;
    late ModelManager manager;

    setUp(() async {
      tmp = TempTestDirectory();
      await tmp.setup();
      manager = ModelManager(modelsPath: tmp.dir.path);
    });

    tearDown(() => tmp.teardown());

    test('deletes an existing model file', () async {
      final file = File('${tmp.dir.path}/model.gguf')
        ..writeAsBytesSync([]);
      expect(file.existsSync(), isTrue);

      await manager.deleteModel('model.gguf');

      expect(file.existsSync(), isFalse);
    });

    test('does nothing when model file does not exist', () async {
      // Should complete without throwing
      await expectLater(
        manager.deleteModel('nonexistent.gguf'),
        completes,
      );
    });

    test('only deletes the specified file, leaving others intact', () async {
      File('${tmp.dir.path}/keep.gguf').writeAsBytesSync([]);
      File('${tmp.dir.path}/remove.gguf').writeAsBytesSync([]);

      await manager.deleteModel('remove.gguf');

      expect(File('${tmp.dir.path}/keep.gguf').existsSync(), isTrue);
      expect(File('${tmp.dir.path}/remove.gguf').existsSync(), isFalse);
    });
  });
}
