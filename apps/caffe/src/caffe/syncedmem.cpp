#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    }
    CUDA_CHECK(cudaFree(gpu_ptr_));
    cudaSetDevice(initial_device);
  }
#endif  // CPU_ONLY
}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaGetDevice(&gpu_device_));
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      own_gpu_data_ = true;
    }
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaGetDevice(&gpu_device_));
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

void SyncedMemory::set_cpu_data(void* data) {
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  own_cpu_data_ = false;
  if (data != NULL) {
    head_ = HEAD_AT_CPU;
  } else {
    head_ = UNINITIALIZED;
  }
}

void SyncedMemory::set_gpu_data(
    void* data, bool change_head, bool allow_reset_cpu_data) {
  // LOG(INFO) << "set_gpu_data for " << this;
  // LOG(INFO) << "data = " << data;
  // LOG(INFO) << "change_head = " << change_head;
  // LOG(INFO) << "own_gpu_data_ = " << own_gpu_data_;
  // LOG(INFO) << "gpu_ptr_ = " << gpu_ptr_;
  // LOG(INFO) << "own_cpu_data_ = " << own_cpu_data_;
  // LOG(INFO) << "cpu_ptr_ = " << cpu_ptr_;
  // LOG(INFO) << "own_cpu_data_ = " << own_cpu_data_;
  // LOG(INFO) << "head_ = " << head_;
  CHECK(!own_gpu_data_) << "own_gpu_data for " << this;
  if (data != NULL) {
    CHECK(!gpu_ptr_);
  }
  gpu_ptr_ = data;
  own_gpu_data_ = false;
  if (change_head) {
    if (data != NULL) {
      CHECK_NE(head_, HEAD_AT_CPU) << ", this = " << this << ", data = " << data;
      head_ = HEAD_AT_GPU;
    } else {
      if (head_ == HEAD_AT_CPU) {
        // LOG(INFO) << "***WARNING: set_gpu_data(): cleaning up CPU data";
        if (!allow_reset_cpu_data) {
          CHECK(0);
        }
      }
      head_ = UNINITIALIZED;
    }
  } else {
    CHECK_NE(head_, HEAD_AT_GPU);
    CHECK_NE(head_, SYNCED);
    if (head_ == UNINITIALIZED) {
      // LOG(INFO) << "***WARNING: set_gpu_data(): no_change_head but no CPU data";
      CHECK(0);
    }
  }
}

void SyncedMemory::set_gpu_data(void* data) {
  CHECK(0) << "This function is not supported";
}

const void* SyncedMemory::cpu_data() {
  to_cpu();
  return (const void*)cpu_ptr_;
}

const void* SyncedMemory::gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

const void* SyncedMemory::check_cpu_data() const {
  return (const void*)cpu_ptr_;
}

const void* SyncedMemory::check_gpu_data() const {
  return (const void*)gpu_ptr_;
}

#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif

}  // namespace caffe

