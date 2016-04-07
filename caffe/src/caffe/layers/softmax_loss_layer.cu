#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/vision_layers.hpp"
#include "cub/cub/cub.cuh"

#define NUM_CUDA_THREADS 128

namespace caffe {

template <typename Dtype, typename Mtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
                                      const Dtype* prob_data, 
                                      const Dtype* label, 
                                      Dtype* loss,
                                      const int num, 
                                      const int dim, 
                                      const int spatial_dim,
                                      const bool has_ignore_label_, 
                                      const int ignore_label_,
                                      Dtype* counts,
                                      Mtype* results) {

  typedef cub::BlockReduce<Mtype, NUM_CUDA_THREADS> BlockReduceF;
  typedef cub::BlockReduce<int,   NUM_CUDA_THREADS> BlockReduceI;

  __shared__ typename BlockReduceF::TempStorage tempStorageF;
  __shared__ typename BlockReduceI::TempStorage tempStorageI;

  Mtype lossSum(0);
  int count(0);
  for( int idx = blockIdx.x*blockDim.x + threadIdx.x ; idx < nthreads ; idx += blockDim.x*gridDim.x ) {
    const int n = idx / spatial_dim;
    const int s = idx % spatial_dim;
    const int label_value = Get<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[idx] = Get<Dtype>(0);
      counts[idx] = Get<Dtype>(0);
    } else {
      Mtype tmp = -log(max(Get<Mtype>(prob_data[n * dim + label_value * spatial_dim + s]), Mtype(FLT_MIN)));
      loss[idx] = Get<Dtype>(tmp);
      counts[idx] = Get<Dtype>(1);
      lossSum += tmp;
      count += 1;
    }
  }

  lossSum = BlockReduceF(tempStorageF).Sum(lossSum);
  count   = BlockReduceI(tempStorageI).Sum(count);

  if( threadIdx.x == 0 ) {
      results[0] = lossSum;
      results[1] = Mtype(count);
  }
}

template< typename Dtype >
struct GetFtype { typedef Dtype Type; };

template<>
struct GetFtype<float16> { typedef float Type; };

template <typename Dtype, typename Mtype>
void SoftmaxWithLossLayer<Dtype,Mtype>::Forward_gpu(
    const vector<Blob<Dtype,Mtype>*>& bottom, const vector<Blob<Dtype,Mtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();

  // TODO: Use 0-copy instead of a memcpy!
  typedef typename GetFtype<Dtype>::Type Ftype;
  Ftype *workspace;
  gpu_memory::allocate((void**) &workspace, 2*sizeof(Ftype));

  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxLossForwardGPU<Dtype, Ftype><<<1, NUM_CUDA_THREADS>>>(
      nthreads, prob_data, label, loss_data,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts, workspace);

  Ftype results[2];
  CUDA_CHECK(cudaMemcpy(results, workspace, sizeof(results), cudaMemcpyDeviceToHost));
  gpu_memory::deallocate(workspace);

  Ftype loss = results[0];
  if (normalize_) {
    loss /= results[1];
  } else {
    loss /= outer_num_;
  }
  top[0]->mutable_cpu_data()[0] = Get<Dtype>(loss);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype, typename Mtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = Get<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = Get<Dtype>(0);
      }
      counts[index] = Get<Dtype>(0);
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] = Get<Dtype>( Get<Mtype>(bottom_diff[n * dim + label_value * spatial_dim + s]) - Get<Mtype>(1) );
      counts[index] = Get<Dtype>(1);
    }
  }
}

template <typename Dtype, typename Mtype>
void SoftmaxWithLossLayer<Dtype,Mtype>::Backward_gpu(const vector<Blob<Dtype,Mtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype,Mtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossBackwardGPU<Dtype,Mtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
    const Mtype loss_weight = Get<Mtype>(top[0]->cpu_diff()[0]);
    if (normalize_) {
      Mtype count;
      caffe_gpu_asum<Dtype,Mtype>(nthreads, counts, &count);
      caffe_gpu_scal<Dtype,Mtype>(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_gpu_scal<Dtype,Mtype>(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe
