#include "third_party/absl/base/call_once.h"
#include "third_party/absl/base/optimization.h"
#include "third_party/absl/status/status.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/stream_executor/tpu/tpu_executor_init_fns.inc"
#include "xla/stream_executor/tpu/tpu_ops_c_api.h"

namespace {

absl::Status DoSetTpuOpsStructFns(void* library_handle) {
  // Initialize a new OpsApiFn struct on the heap.
  auto* new_fns = new TfTpu_OpsApiFn();

  TFTPU_SET_FN(new_fns, ConfigureDistributedTpuOp_DoWork);
  TFTPU_SET_FN(new_fns, WaitForDistributedTpuOp_DoWork);
  TFTPU_SET_FN(new_fns, InitializeHostForDistributedTpuOp_DoWork);
  TFTPU_SET_FN(new_fns, SetGlobalTPUArrayOp_DoWork);
  TFTPU_SET_FN(new_fns, DisconnectDistributedTpuChipsOp_DoWork);
  TFTPU_SET_FN(new_fns, TpuConfigurationApi_FreeCharArray);
  TFTPU_SET_FN(new_fns, TpuConfigurationApi_FreeInt32Array);
  TFTPU_SET_FN(new_fns, TpuConfigurationApi_HasTPUPodState);
  TFTPU_SET_FN(new_fns, TpuConfigurationApi_TpusPerHost);
  TFTPU_SET_FN(new_fns, TpuConfigurationApi_TpuMemoryLimit);
  TFTPU_SET_FN(new_fns, TpuConfigurationApi_RemoteCompilationCacheSizeInBytes);
  TFTPU_SET_FN(new_fns,
               TpuConfigurationApi_CompilationCacheServerAddressFromConfig);
  TFTPU_SET_FN(new_fns, TpuConfigurationApi_GetServerAddressAndPort);

  TFTPU_SET_FN(new_fns, TpuMeshState_Create);
  TFTPU_SET_FN(new_fns, TpuMeshState_Free);
  TFTPU_SET_FN(new_fns, TpuMeshState_MeshCommonState);

  TFTPU_SET_FN(new_fns, TpuEmbeddingEngineState_Create);
  TFTPU_SET_FN(new_fns, TpuEmbeddingEngineState_Free);
  TFTPU_SET_FN(new_fns, TpuEmbeddingEngineState_GetState);

  TFTPU_SET_FN(new_fns, TpuCompile_CompileAndBuild);

  TFTPU_SET_FN(new_fns, TpuExecutable_LoadProgramAndEnqueueToStream);

  TFTPU_SET_FN(new_fns, HardwareLayout_HostShapeToDeviceShape);
  TFTPU_SET_FN(new_fns, HardwareLayout_ShapeSize);
  TFTPU_SET_FN(new_fns, HardwareLayout_ShapeSizeCompact);
  TFTPU_SET_FN(new_fns, HardwareLayout_ShapeSizeCompactRaw);

  TFTPU_SET_FN(new_fns, TpuExecute_RuntimeInputToPaddedData);

  TFTPU_SET_FN(new_fns, TpuExecute_GetTpuEmbeddingMemoryAllocations);
  TFTPU_SET_FN(new_fns, TpuExecute_FreeTpuEmbeddingMemoryAllocations);

  TFTPU_SET_FN(new_fns, TpuProgram_New);
  TFTPU_SET_FN(new_fns, TpuProgram_Free);
  TFTPU_SET_FN(new_fns, TpuProgram_NewArray);
  TFTPU_SET_FN(new_fns, TpuProgram_FreeArray);
  TFTPU_SET_FN(new_fns, TpuProgram_UnloadAndDestroy);
  TFTPU_SET_FN(new_fns, TpuProgram_GetProgramSize);
  TFTPU_SET_FN(new_fns, TpuProgram_LogProgramMemorySummary);
  TFTPU_SET_FN(new_fns, TpuProgram_GetExecutableInfo);
  TFTPU_SET_FN(new_fns, TpuProgram_GetHostTransferInfo);
  TFTPU_SET_FN(new_fns, TpuProgram_GetHloMetadata);
  TFTPU_SET_FN(new_fns, TpuProgram_GetMayModifyVariables);
  TFTPU_SET_FN(new_fns, TpuProgram_HasSharding);
  TFTPU_SET_FN(new_fns, TpuProgram_GetTpuProgram);
  TFTPU_SET_FN(new_fns, TpuProgram_SerializeTpuExecutable);
  TFTPU_SET_FN(new_fns, TpuProgram_SerializeCompilerMetadata);
  TFTPU_SET_FN(new_fns, TpuProgram_DeserializeFromGetTpuProgramResponseProto);
  TFTPU_SET_FN(new_fns, TpuProgram_GetFingerprint);
  TFTPU_SET_FN(new_fns, TpuProgram_DestroyFingerprint);

  TFTPU_SET_FN(new_fns, TpuNodeContext_Create);
  TFTPU_SET_FN(new_fns, TpuNodeContext_Free);
  TFTPU_SET_FN(new_fns, TpuNodeContext_Initialize);
  TFTPU_SET_FN(new_fns, TpuNodeContext_CloseTpuHost);
  TFTPU_SET_FN(new_fns, TpuNodeContext_CompactionSupported);

  TFTPU_SET_FN(new_fns, TpuTopology_AvailableCoreCount);
  TFTPU_SET_FN(new_fns, TpuTopology_AvailableCoresPerChip);
  TFTPU_SET_FN(new_fns, TpuTopology_MaybeAvailableSparseCoresPerLogicalDevice);
  TFTPU_SET_FN(new_fns, TpuNetUtil_RecycleUnusedPort);
  TFTPU_SET_FN(new_fns, TpuCompile_IsTpuCompilationEnabled);
  TFTPU_SET_FN(new_fns, TpuCompile_ShouldTpuCompileOpIgnoreCancellation);
  TFTPU_SET_FN(new_fns, TpuCompile_CreateCompilationCacheKey);
  TFTPU_SET_FN(new_fns, TpuCompile_DestroyCompilationCacheKey);
  TFTPU_SET_FN(new_fns, TpuCompile_CreateGuaranteedConstFingerprint);
  TFTPU_SET_FN(new_fns, TpuUtil_GetTopologyPtr);
  TFTPU_SET_FN(new_fns, TpuUtil_GetXlaPadSizeFromTpuTopology);

  TFTPU_SET_FN(new_fns, TfTpu_InitializeTpuModelServer);

  TFTPU_SET_FN(new_fns, TfTpuOrdinalSelector_Create);
  TFTPU_SET_FN(new_fns, TfTpuOrdinalSelector_Destroy);
  TFTPU_SET_FN(new_fns, TfTpuOrdinalSelector_GetOrdinal);
  TFTPU_SET_FN(new_fns, TfTpuOrdinalSelector_DequeueFromCoreSelector);
  TFTPU_SET_FN(new_fns, TfTpu_GetTpuPartitionedCallParams);

  TFTPU_SET_FN(new_fns, TpuEmbeddingEngine_ExecutePartitioner);
  TFTPU_SET_FN(new_fns, TpuEmbeddingEngine_ConfigureMemory);
  TFTPU_SET_FN(new_fns, TpuEmbeddingEngine_CollateMemory);
  TFTPU_SET_FN(new_fns, TpuEmbeddingEngine_ConfigureHost);
  TFTPU_SET_FN(new_fns, TpuEmbeddingEngine_ConnectHosts);
  TFTPU_SET_FN(new_fns, TpuEmbeddingEngine_Finalize);
  TFTPU_SET_FN(new_fns, TpuEmbeddingEngine_IsInitialized);
  TFTPU_SET_FN(new_fns, TpuEmbeddingEngine_WriteParameters);
  TFTPU_SET_FN(new_fns, TpuEmbeddingEngine_ReadParameters);
  TFTPU_SET_FN(new_fns, TpuEmbeddingTensorBatchFixedState_Create);
  TFTPU_SET_FN(new_fns, TpuEmbeddingTensorBatchFixedState_Destroy);
  TFTPU_SET_FN(new_fns, TpuEmbeddingEngine_EnqueueTensorBatch);
  TFTPU_SET_FN(new_fns, TpuEmbeddingEngine_RecvActivationsComputation);
  TFTPU_SET_FN(new_fns,
               TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation);
  TFTPU_SET_FN(new_fns,
               TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation);
  TFTPU_SET_FN(new_fns, TpuEmbeddingEngine_DedupDataSizeComputation);
  TFTPU_SET_FN(new_fns, TpuEmbeddingEngine_DedupDataTupleMaskComputation);

  TFTPU_SET_FN(new_fns, SparseCore_GetMaxIdsAndUniques);

  // Atomically publish the fully initialized OpsApiFn struct.
  auto* ops_api_fn =
      const_cast<TfTpu_OpsApiFn*>(::stream_executor::tpu::OpsApiFn());
  void* compile_and_build_fn =
      reinterpret_cast<void*>(new_fns->TpuCompile_CompileAndBuildFn);
  new_fns->TpuCompile_CompileAndBuildFn = nullptr;
  size_t offset = offsetof(TfTpu_OpsApiFn, TpuMeshState_CreateFn);
  memcpy(reinterpret_cast<char*>(ops_api_fn) + offset,
         reinterpret_cast<char*>(new_fns) + offset,
         sizeof(TfTpu_OpsApiFn) - offset);
  reinterpret_cast<std::atomic<void*>*>(
      &ops_api_fn->TpuCompile_CompileAndBuildFn)
      ->store(compile_and_build_fn,
              std::memory_order_release);
  delete new_fns;

  *::stream_executor::tpu::g_tpu_ops_struct_fns_init_status = absl::OkStatus();
  return absl::OkStatus();
}

absl::Status SetTpuOpsStructFns(void* library_handle) {  // TENSORFLOW_STATUS_OK
  absl::call_once(::stream_executor::tpu::g_tpu_ops_struct_fns_once,
                  [library_handle]() { DoSetTpuOpsStructFns(library_handle).IgnoreError(); });
  return *::stream_executor::tpu::g_tpu_ops_struct_fns_init_status;
}

absl::Status InitializeTpuStructFns(  // TENSORFLOW_STATUS_OK
    void* library_handle) {
  TF_RETURN_IF_ERROR(SetTpuOpsStructFns(library_handle));
  TF_RETURN_IF_ERROR(SetExecutorStructFn(library_handle));

  return absl::OkStatus();
}

}  // namespace
