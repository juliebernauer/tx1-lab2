#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void ReLULayer<Dtype,Mtype>::Forward_cpu(const vector<Blob<Dtype,Mtype>*>& bottom,
    const vector<Blob<Dtype,Mtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Mtype negative_slope(this->layer_param_.relu_param().negative_slope());
  for (int i = 0; i < count; ++i) {
    top_data[i] = Get<Dtype>( std::max(Get<Mtype>(bottom_data[i]), Mtype(0))
        + negative_slope * std::min(Get<Mtype>(bottom_data[i]), Mtype(0)) );
  }
}

template <typename Dtype, typename Mtype>
void ReLULayer<Dtype,Mtype>::Backward_cpu(const vector<Blob<Dtype,Mtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype,Mtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Mtype negative_slope(this->layer_param_.relu_param().negative_slope());
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = Get<Dtype>( Get<Mtype>(top_diff[i]) * ((Get<Mtype>(bottom_data[i]) > 0)
          + negative_slope * (Get<Mtype>(bottom_data[i]) <= 0)) );
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
