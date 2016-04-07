#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

const float kBNLL_THRESHOLD = 50.;

template <typename Dtype, typename Mtype>
void BNLLLayer<Dtype,Mtype>::Forward_cpu(const vector<Blob<Dtype,Mtype>*>& bottom,
    const vector<Blob<Dtype,Mtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = Get<Dtype>( Get<Mtype>(bottom_data[i]) > 0.f ?
        Get<Mtype>(bottom_data[i]) + log(1. + exp(-Get<Mtype>(bottom_data[i]))) :
        log(1. + exp(Get<Mtype>(bottom_data[i]))) );
  }
}

template <typename Dtype, typename Mtype>
void BNLLLayer<Dtype,Mtype>::Backward_cpu(const vector<Blob<Dtype,Mtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype,Mtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Mtype expval;
    for (int i = 0; i < count; ++i) {
      expval = exp(std::min(Get<Mtype>(bottom_data[i]), Mtype(kBNLL_THRESHOLD)));
      bottom_diff[i] = Get<Dtype>( Get<Mtype>(top_diff[i]) * expval / (expval + 1.) );
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BNLLLayer);
#endif

INSTANTIATE_CLASS(BNLLLayer);
REGISTER_LAYER_CLASS(BNLL);

}  // namespace caffe
