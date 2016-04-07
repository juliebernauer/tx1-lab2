#include <boost/thread.hpp>
#include "caffe/layer.hpp"

namespace caffe {

void LayerBase::InitMutex() {
  forward_mutex_.reset(new boost::mutex());
}


void LayerBase::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

void LayerBase::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}

INSTANTIATE_CLASS(Layer);

}  // namespace caffe
