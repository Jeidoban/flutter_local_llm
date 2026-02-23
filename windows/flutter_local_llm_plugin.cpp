#include "flutter_local_llm_plugin.h"

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>

#include <memory>

namespace flutter_local_llm {

// static
void FlutterLocalLlmPlugin::RegisterWithRegistrar(
    flutter::PluginRegistrarWindows *registrar) {
  auto channel =
      std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
          registrar->messenger(), "flutter_local_llm",
          &flutter::StandardMethodCodec::GetInstance());

  auto plugin = std::make_unique<FlutterLocalLlmPlugin>();

  channel->SetMethodCallHandler(
      [plugin_pointer = plugin.get()](const auto &call, auto result) {
        plugin_pointer->HandleMethodCall(call, std::move(result));
      });

  registrar->AddPlugin(std::move(plugin));
}

FlutterLocalLlmPlugin::FlutterLocalLlmPlugin() {}

FlutterLocalLlmPlugin::~FlutterLocalLlmPlugin() {}

void FlutterLocalLlmPlugin::HandleMethodCall(
    const flutter::MethodCall<flutter::EncodableValue> &method_call,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  result->NotImplemented();
}

}  // namespace flutter_local_llm
