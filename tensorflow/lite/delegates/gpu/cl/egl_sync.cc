/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/gpu/cl/egl_sync.h"

#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"

namespace tflite {
namespace gpu {
namespace cl {

namespace {

bool IsExtensionSupported(EGLDisplay display, const char* extension) {
  const char* extensions = eglQueryString(display, EGL_EXTENSIONS);
  if (extensions && std::strstr(extensions, extension)) {
    return true;
  }
  return false;
}

absl::Status IsEglFenceSyncSupported(EGLDisplay display) {
  static bool supported = IsExtensionSupported(display, "EGL_KHR_fence_sync");
  if (!supported) {
    return absl::InternalError("Not supported: EGL_KHR_fence_sync");
  }
  return absl::OkStatus();
}

absl::Status IsEglWaitSyncSupported(EGLDisplay display) {
  static bool supported = IsExtensionSupported(display, "EGL_KHR_wait_sync");
  if (!supported) {
    return absl::InternalError("Not supported: EGL_KHR_wait_sync");
  }
  return absl::OkStatus();
}

} // anonymous namespace

absl::Status EglSync::NewFence(EGLDisplay display, EglSync* sync) {
  static auto* egl_create_sync_khr =
      reinterpret_cast<decltype(&eglCreateSyncKHR)>(
          eglGetProcAddress("eglCreateSyncKHR"));
  // Needs extension: EGL_KHR_fence_sync (EGL) / GL_OES_EGL_sync (OpenGL ES).
  RETURN_IF_ERROR(IsEglFenceSyncSupported(display));
  EGLSyncKHR egl_sync;
  RETURN_IF_ERROR(TFLITE_GPU_CALL_EGL(*egl_create_sync_khr, &egl_sync, display,
                                      EGL_SYNC_FENCE_KHR, nullptr));
  if (egl_sync == EGL_NO_SYNC_KHR) {
    return absl::InternalError("Returned empty KHR EGL sync");
  }
  *sync = EglSync(display, egl_sync);
  return absl::OkStatus();
}

EglSync& EglSync::operator=(EglSync&& sync) {
  if (this != &sync) {
    Invalidate();
    std::swap(sync_, sync.sync_);
    display_ = sync.display_;
  }
  return *this;
}

void EglSync::Invalidate() {
  if (sync_ != EGL_NO_SYNC_KHR) {
    static auto* egl_destroy_sync_khr =
        reinterpret_cast<decltype(&eglDestroySyncKHR)>(
            eglGetProcAddress("eglDestroySyncKHR"));
    // Needs extension: EGL_KHR_fence_sync (EGL) / GL_OES_EGL_sync (OpenGL ES).
    if (IsEglFenceSyncSupported(display_).ok()) {
      (*egl_destroy_sync_khr)(display_, sync_);
    }
    sync_ = EGL_NO_SYNC_KHR;
  }
}

absl::Status EglSync::ServerWait() {
  static auto* egl_wait_sync_khr = reinterpret_cast<decltype(&eglWaitSyncKHR)>(
      eglGetProcAddress("eglWaitSyncKHR"));
  // Needs extension: EGL_KHR_wait_sync
  RETURN_IF_ERROR(IsEglWaitSyncSupported(display_));
  EGLint result;
  RETURN_IF_ERROR(
      TFLITE_GPU_CALL_EGL(*egl_wait_sync_khr, &result, display_, sync_, 0));
  return result == EGL_TRUE ? absl::OkStatus()
                            : absl::InternalError("eglWaitSync failed");
}

absl::Status EglSync::ClientWait() {
  static auto* egl_client_wait_sync_khr =
      reinterpret_cast<decltype(&eglClientWaitSyncKHR)>(
          eglGetProcAddress("eglClientWaitSyncKHR"));
  // Needs extension: EGL_KHR_fence_sync (EGL) / GL_OES_EGL_sync (OpenGL ES).
  RETURN_IF_ERROR(IsEglFenceSyncSupported(display_));
  EGLint result;
  // TODO(akulik): make it active wait for better performance
  RETURN_IF_ERROR(
      TFLITE_GPU_CALL_EGL(*egl_client_wait_sync_khr, &result, display_, sync_,
                          EGL_SYNC_FLUSH_COMMANDS_BIT_KHR, EGL_FOREVER_KHR));
  return result == EGL_CONDITION_SATISFIED_KHR
             ? absl::OkStatus()
             : absl::InternalError("eglClientWaitSync failed");
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
