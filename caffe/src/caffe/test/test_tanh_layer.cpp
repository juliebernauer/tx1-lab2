#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

double tanh_naive(double x) {
  if (x < -40) {
    // avoid negative overflow
    return -1;
  } else if (x > 40) {
    // avoid positive overflow
    return 1;
  } else {
    // exact expression for tanh, which is unstable for large x
    double exp2x = exp(2 * x);
    return (exp2x - 1.0) / (exp2x + 1.0);
  }
}

template <typename TypeParam>
class TanHLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;

 protected:
  TanHLayerTest()
      : blob_bottom_(new Blob<Dtype,Mtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype,Mtype>()) {
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~TanHLayerTest() { delete blob_bottom_; delete blob_top_; }

  void TestForward(Mtype filler_std) {
    FillerParameter filler_param;
    filler_param.set_std(filler_std);
    GaussianFiller<Dtype,Mtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    LayerParameter layer_param;
    TanHLayer<Dtype,Mtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const Dtype min_precision = Get<Dtype>(1e-5);
    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
      Mtype expected_value = tanh_naive(Get<Mtype>(bottom_data[i]));
      Mtype precision = std::max(
        Mtype(std::abs(expected_value * Mtype(1e-4))), Get<Mtype>(min_precision));
      EXPECT_NEAR(expected_value, Get<Mtype>(top_data[i]), tol<Dtype>(precision));
    }
  }

  void TestBackward(Mtype filler_std) {
    FillerParameter filler_param;
    filler_param.set_std(filler_std);
    GaussianFiller<Dtype,Mtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    LayerParameter layer_param;
    TanHLayer<Dtype,Mtype> layer(layer_param);
    GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-2), 1701);
    checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }

  Blob<Dtype,Mtype>* const blob_bottom_;
  Blob<Dtype,Mtype>* const blob_top_;
  vector<Blob<Dtype,Mtype>*> blob_bottom_vec_;
  vector<Blob<Dtype,Mtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TanHLayerTest, TestDtypesAndDevices);

TYPED_TEST(TanHLayerTest, TestTanH) {
  this->TestForward(1.0);
}

TYPED_TEST(TanHLayerTest, TestTanHOverflow) {
  // this will fail if tanh overflow is not properly handled
  this->TestForward(10000.0);
}

TYPED_TEST(TanHLayerTest, TestTanHGradient) {
  this->TestBackward(1.0);
}

}  // namespace caffe
