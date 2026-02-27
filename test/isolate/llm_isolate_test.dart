import 'dart:isolate';
import 'package:flutter_local_llm/src/isolate/llm_isolate.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  // ============================================================
  // Command types
  // ============================================================

  group('Command types', () {
    test('GenerateFromPromptCommand stores all fields', () {
      final cmd = GenerateFromPromptCommand(
        prompt: 'Hello',
        requestId: 3,
        attachmentPaths: ['/img.jpg'],
      );
      expect(cmd.prompt, 'Hello');
      expect(cmd.requestId, 3);
      expect(cmd.attachmentPaths, ['/img.jpg']);
    });

    test('GenerateFromPromptCommand attachmentPaths defaults to null', () {
      final cmd = GenerateFromPromptCommand(prompt: 'Hi', requestId: 0);
      expect(cmd.attachmentPaths, isNull);
    });

    test('GetRemainingContextCommand stores requestId', () {
      final cmd = GetRemainingContextCommand(requestId: 7);
      expect(cmd.requestId, 7);
    });
  });

  // ============================================================
  // Response types
  // ============================================================

  group('Response types', () {
    test('TokenResponse stores token and requestId', () {
      final r = TokenResponse(token: 'hello', requestId: 1);
      expect(r.token, 'hello');
      expect(r.requestId, 1);
    });

    test('CompletionResponse stores requestId', () {
      final r = CompletionResponse(requestId: 2);
      expect(r.requestId, 2);
    });

    test('RemainingContextResponse stores remaining and requestId', () {
      final r = RemainingContextResponse(remaining: 2048, requestId: 4);
      expect(r.remaining, 2048);
      expect(r.requestId, 4);
    });

    test('ErrorResponse requestId is null by default', () {
      final r = ErrorResponse(error: 'something went wrong');
      expect(r.error, 'something went wrong');
      expect(r.requestId, isNull);
    });

    test('ErrorResponse requestId can be set for generate errors', () {
      final r = ErrorResponse(error: 'gen error', requestId: 5);
      expect(r.requestId, 5);
    });
  });

  // ============================================================
  // LlamaManager error handling
  //
  // LlamaManager has a public constructor that accepts a SendPort,
  // so we can construct it with a real ReceivePort and verify the
  // catch block sends the correct ErrorResponse when commands are
  // received before the model is initialized (_llama is late).
  // ============================================================

  group('LlamaManager error handling', () {
    late ReceivePort receivePort;
    late LlamaManager manager;

    setUp(() {
      receivePort = ReceivePort();
      manager = LlamaManager(receivePort.sendPort);
    });

    tearDown(() {
      receivePort.close();
    });

    test(
      'GenerateFromPromptCommand sends ErrorResponse with matching requestId',
      () async {
        final responseFuture = receivePort.first;
        await manager.handleCommand(
          GenerateFromPromptCommand(prompt: 'test', requestId: 42),
        );
        final response = await responseFuture;

        expect(response, isA<ErrorResponse>());
        expect((response as ErrorResponse).requestId, 42);
      },
    );

    test(
      'GetRemainingContextCommand sends ErrorResponse with null requestId',
      () async {
        // requestId is only forwarded for GenerateFromPromptCommand in the catch block
        final responseFuture = receivePort.first;
        await manager.handleCommand(GetRemainingContextCommand(requestId: 7));
        final response = await responseFuture;

        expect(response, isA<ErrorResponse>());
        expect((response as ErrorResponse).requestId, isNull);
      },
    );

    test(
      'ClearContextCommand sends ErrorResponse with null requestId',
      () async {
        final responseFuture = receivePort.first;
        await manager.handleCommand(ClearContextCommand());
        final response = await responseFuture;

        expect(response, isA<ErrorResponse>());
        expect((response as ErrorResponse).requestId, isNull);
      },
    );

    test('ErrorResponse error message is non-empty', () async {
      final responseFuture = receivePort.first;
      await manager.handleCommand(ClearContextCommand());
      final response = await responseFuture as ErrorResponse;

      expect(response.error, isNotEmpty);
    });
  });
}
