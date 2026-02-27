import 'dart:io';
import 'package:flutter_local_llm/src/managers/model_manager.dart';
import 'package:flutter_test/flutter_test.dart';
import '../test_helpers.dart';

void main() {
  group('ModelManager constructor', () {
    test('stores modelsPath and onDownloadProgress', () {
      var callCount = 0;
      final manager = ModelManager(
        modelsPath: '/some/path',
        onDownloadProgress: (_) => callCount++,
      );

      expect(manager.modelsPath, '/some/path');
      manager.onDownloadProgress?.call(0.5);
      expect(callCount, 1);
    });

    test('allows null modelsPath and callback', () {
      final manager = ModelManager();
      expect(manager.modelsPath, isNull);
      expect(manager.onDownloadProgress, isNull);
    });
  });

  group('ModelManager getModelPath()', () {
    late TempTestDirectory tempDir;

    setUp(() async {
      tempDir = TempTestDirectory();
      await tempDir.setup();
    });

    tearDown(() async {
      await tempDir.teardown();
    });

    test('returns path to existing file without downloading', () async {
      final manager = ModelManager(modelsPath: tempDir.dir.path);
      final fakeModel = File('${tempDir.dir.path}/my-model.gguf');
      fakeModel.writeAsStringSync('fake model data');

      final path = await manager.getModelPath('my-model.gguf');

      expect(path, fakeModel.path);
      expect(File(path).existsSync(), true);
    });

    test('returns path string even when file does not exist and no URL given',
        () async {
      final manager = ModelManager(modelsPath: tempDir.dir.path);

      final path = await manager.getModelPath('missing-model.gguf');

      expect(path, endsWith('missing-model.gguf'));
      expect(File(path).existsSync(), false);
    });

    test('creates modelsPath directory if it does not exist', () async {
      final nestedPath = '${tempDir.dir.path}/a/b/c';
      final manager = ModelManager(modelsPath: nestedPath);
      final fakeModel = File('$nestedPath/model.gguf');
      fakeModel.createSync(recursive: true);

      final path = await manager.getModelPath('model.gguf');

      expect(Directory(nestedPath).existsSync(), true);
      expect(path, endsWith('model.gguf'));
    });

    test('does not call onDownloadProgress when file already exists', () async {
      var progressCalled = false;
      final manager = ModelManager(
        modelsPath: tempDir.dir.path,
        onDownloadProgress: (_) => progressCalled = true,
      );
      File('${tempDir.dir.path}/existing.gguf').writeAsStringSync('data');

      await manager.getModelPath('existing.gguf', downloadUrl: 'https://example.com/model.gguf');

      expect(progressCalled, false);
    });
  });
}
