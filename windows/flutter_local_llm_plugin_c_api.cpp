#include "include/flutter_local_llm/flutter_local_llm_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "flutter_local_llm_plugin.h"

void FlutterLocalLlmPluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  flutter_local_llm::FlutterLocalLlmPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
