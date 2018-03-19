#include <functional>

#include <iomanip>
#include <sstream>

#include "caffe2/core/context_gpu.h"
#include "stat_op.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void XYpZ(const int nthreads, const T* Xdata, const T* Ydata,
                     T* Zdata) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    Zdata[index] = Xdata[index] * Ydata[index] + Zdata[index];
  }
}

}  // namespace

template <>
bool StatOp<float, CUDAContext>::RunOnDevice() {
  const int gpu_id = context_.device_id();
  if (gpu_id == 0) {
  } else {
    return true;
  }
  const auto& I = Input(0);
  const auto& L = Input(1);

  auto* AI = Output(0);
  auto* AL = Output(1);

  if (init_) {
    AI->ResizeLike(I);
    math::Set<float, CUDAContext>(AI->numel(), 0.f, AI->mutable_data<float>(),
                                  &context_);
    AL->ResizeLike(L);
    math::Set<float, CUDAContext>(AL->numel(), 0.f, AL->mutable_data<float>(),
                                  &context_);
    init_ = false;
  }

  XYpZ<float><<<CAFFE_GET_BLOCKS(I.numel()), CAFFE_CUDA_NUM_THREADS, 0,
                context_.cuda_stream()>>>(
      I.numel(), I.data<float>(), L.data<float>(), AI->mutable_data<float>());

  math::Add<float, CUDAContext>(L.numel(), L.data<float>(), AL->data<float>(),
                                AL->mutable_data<float>(), &context_);

  cur_iter_++;
  if (cur_iter_ % display_ == 0 || cur_iter_ == 1) {
    Tensor AI_cpu = Tensor(caffe2::CPU);
    AI_cpu.ResizeLike(*AI);

    Tensor AL_cpu = Tensor(caffe2::CPU);
    AL_cpu.ResizeLike(*AL);

    AI_cpu.CopyFrom(*AI, false);
    AL_cpu.CopyFrom(*AL, false);
    context_.FinishDeviceComputation();

    const float* AI_cpu_data = AI_cpu.data<float>();
    const float* AL_cpu_data = AL_cpu.data<float>();

    std::stringstream stream;
    stream << "\t" + prefix_ + " Stat #iter_: " + std::to_string(cur_iter_);
    for (int i = 0; i < AI_cpu.numel(); i++) {
      stream << " " << std::fixed << std::setprecision(2)
             << (AI_cpu_data[i] / AL_cpu_data[i]);
    }
    std::cout << stream.str() << std::endl;
    init_ = true;
  }
  return true;
}

REGISTER_CUDA_OPERATOR(Stat, StatOp<float, CUDAContext>);

}  // namespace caffe2
