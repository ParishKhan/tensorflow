#include <atomic>
#include <cstddef>
#include <cstring>

#include "third_party/absl/base/call_once.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/stream_executor/tpu/tpu_profiler_c_api.h"

namespace {

absl::Status SetTpuProfilerApiFns(
    void* library_handle) {  // TENSORFLOW_STATUS_OK
  absl::call_once(
      ::stream_executor::tpu::g_tpu_profiler_init_once, [library_handle]() {
        // Constant cast so that we can initialize the functions. The functions
        // are mutable here because this is the only place where they are
        // initialized.
        auto* profiler_api_fn = const_cast<TfTpu_ProfilerApiFn*>(
            ::stream_executor::tpu::ProfilerApiFn());
        TfTpu_ProfilerApiFn temp_obj = {};
        TfTpu_ProfilerApiFn* temp = &temp_obj;
        TFTPU_SET_FN(temp, TpuProfiler_Create);
        TFTPU_SET_FN(temp, TpuProfiler_Destroy);
        TFTPU_SET_FN(temp, TpuProfiler_Start);
        TFTPU_SET_FN(temp, TpuProfiler_Stop);
        TFTPU_SET_FN(temp, TpuProfiler_CollectData);

        TFTPU_SET_FN(temp, TpuStatus_New);
        TFTPU_SET_FN(temp, TpuStatus_Free);
        TFTPU_SET_FN(temp, TpuStatus_Message);
        TFTPU_SET_FN(temp, TpuStatus_Code);

        void* create_fn = reinterpret_cast<void*>(temp->TpuProfiler_CreateFn);
        temp->TpuProfiler_CreateFn = nullptr;
        size_t offset = offsetof(TfTpu_ProfilerApiFn, TpuProfiler_DestroyFn);
        memcpy(reinterpret_cast<char*>(profiler_api_fn) + offset,
               reinterpret_cast<char*>(temp) + offset,
               sizeof(TfTpu_ProfilerApiFn) - offset);
        reinterpret_cast<std::atomic<void*>*>(
            &profiler_api_fn->TpuProfiler_CreateFn)
            ->store(create_fn, std::memory_order_release);
      });

  return absl::OkStatus();
}

}  // namespace
