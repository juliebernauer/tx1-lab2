#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void ContrastiveLossLayer<Dtype,Mtype>::Forward_gpu(
    const vector<Blob<Dtype,Mtype>*>& bottom, const vector<Blob<Dtype,Mtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_gpu_sub<Dtype,Mtype>(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[1]->gpu_data(),  // b
      diff_.mutable_gpu_data());  // a_i-b_i
  caffe_gpu_powx<Dtype,Mtype>(
      count,
      diff_.mutable_gpu_data(),  // a_i-b_i
      Mtype(2),
      diff_sq_.mutable_gpu_data());  // (a_i-b_i)^2
  caffe_gpu_gemv<Dtype,Mtype>(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Mtype(1.0),
      diff_sq_.gpu_data(),  // (a_i-b_i)^2
      summer_vec_.gpu_data(),
      Mtype(0.0),
      dist_sq_.mutable_gpu_data());  // \Sum (a_i-b_i)^2
  Mtype margin(this->layer_param_.contrastive_loss_param().margin());
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  Mtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    if (Get<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
      loss += Get<Mtype>(dist_sq_.cpu_data()[i]);
    } else {  // dissimilar pairs
      if (legacy_version) {
        loss += std::max((float)(margin - dist_sq_.cpu_data()[i]), 0.F);
      } else {
        Mtype dist = std::max((float)(margin - sqrt(dist_sq_.cpu_data()[i])), 0.F);
        loss += dist*dist;
      }
    }
  }
  loss = loss / static_cast<Mtype>(bottom[0]->num()) / Mtype(2);
  top[0]->mutable_cpu_data()[0] = Get<Dtype>(loss);
}

template <typename Dtype, typename Mtype>
__global__ void CLLBackward(const int count, const int channels,
    const Mtype margin, const bool legacy_version, const Mtype alpha,
    const Dtype* y, const Dtype* diff, const Dtype* dist_sq,
    Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    int n = i / channels;  // the num index, to access y and dist_sq
    if (Get<int>(y[n])) {  // similar pairs
      bottom_diff[i] = Get<Dtype>( alpha * Get<Mtype>(diff[i]) );
    } else {  // dissimilar pairs
      Mtype mdist(0.0);
      Mtype beta(0.0);
      if (legacy_version) {
        mdist = (margin - Get<Mtype>(dist_sq[n]));
        beta = -alpha;
      } else {
        Mtype dist(sqrt(Get<Mtype>(dist_sq[n])));
        mdist = (margin - dist);
        beta = -alpha * mdist / (dist + Mtype(1e-4)) * Get<Mtype>(diff[i]);
      }
      if (mdist > 0.0) {
        bottom_diff[i] = Get<Dtype>(beta);
      } else {
        bottom_diff[i] = Get<Dtype>(0);
      }
    }
  }
}

template <typename Dtype, typename Mtype>
void ContrastiveLossLayer<Dtype,Mtype>::Backward_gpu(const vector<Blob<Dtype,Mtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype,Mtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const int count = bottom[0]->count();
      const int channels = bottom[0]->channels();
      Mtype margin(this->layer_param_.contrastive_loss_param().margin());
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
      const Mtype sign(i == 0 ? 1 : -1);
      const Mtype alpha(sign * Get<Mtype>(top[0]->cpu_diff()[0]) /
          static_cast<Mtype>(bottom[0]->num()));
      // NOLINT_NEXT_LINE(whitespace/operators)
      CLLBackward<Dtype,Mtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha,
          bottom[2]->gpu_data(),  // pair similarity 0 or 1
          diff_.gpu_data(),  // the cached eltwise difference between a and b
          dist_sq_.gpu_data(),  // the cached square distance between a and b
          bottom[i]->mutable_gpu_diff());
      CUDA_POST_KERNEL_CHECK;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ContrastiveLossLayer);

}  // namespace caffe
