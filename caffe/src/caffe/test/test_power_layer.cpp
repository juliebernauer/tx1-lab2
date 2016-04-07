#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class PowerLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;

 protected:
  PowerLayerTest()
      : blob_bottom_(new Blob<Dtype,Mtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype,Mtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype,Mtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~PowerLayerTest() { delete blob_bottom_; delete blob_top_; }

  void TestForward(Mtype power, Mtype scale, Mtype shift) {
    LayerParameter layer_param;
    layer_param.mutable_power_param()->set_power(power);
    layer_param.mutable_power_param()->set_scale(scale);
    layer_param.mutable_power_param()->set_shift(shift);
    PowerLayer<Dtype,Mtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const Mtype min_precision = 1e-5;
    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
      Mtype expected_value = pow(shift + scale * Get<Mtype>(bottom_data[i]), power);
      if (power == Mtype(0) || power == Mtype(1) || power == Mtype(2)) {
        EXPECT_FALSE(isnan(top_data[i]));
      }
      if (isnan(expected_value)) {
        EXPECT_TRUE(isnan(top_data[i]));
      } else {
        Mtype precision = std::max(
          Mtype(std::abs(expected_value * Mtype(1e-4))), min_precision);
        EXPECT_NEAR(expected_value, Get<Mtype>(top_data[i]), tol<Dtype>(precision));
      }
    }
  }

  void TestBackward(Mtype power, Mtype scale, Mtype shift) {
    LayerParameter layer_param;
    layer_param.mutable_power_param()->set_power(power);
    layer_param.mutable_power_param()->set_scale(scale);
    layer_param.mutable_power_param()->set_shift(shift);
    PowerLayer<Dtype,Mtype> layer(layer_param);
    if (power != Mtype(0) && power != Mtype(1) && power != Mtype(2)) {
      // Avoid NaNs by forcing (shift + scale * x) >= 0
      Dtype* bottom_data = this->blob_bottom_->mutable_cpu_data();
      Mtype min_value = -shift / scale;
      for (int i = 0; i < this->blob_bottom_->count(); ++i) {
        if (Get<Mtype>(bottom_data[i]) < min_value) {
          bottom_data[i] = Get<Dtype>(min_value + (min_value - Get<Mtype>(bottom_data[i])));
        }
      }
    }
    GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-3), Get<Dtype>(1e-2), 1701, Get<Dtype>(0.), Get<Dtype>(0.01));
    checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }

  Blob<Dtype,Mtype>* const blob_bottom_;
  Blob<Dtype,Mtype>* const blob_top_;
  vector<Blob<Dtype,Mtype>*> blob_bottom_vec_;
  vector<Blob<Dtype,Mtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PowerLayerTest, TestDtypesAndDevices);

TYPED_TEST(PowerLayerTest, TestPower) {
  typedef typename TypeParam::Mtype Mtype;
  Mtype power = 0.37;
  Mtype scale = 0.83;
  Mtype shift = -2.4;
  this->TestForward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerGradient) {
  typedef typename TypeParam::Mtype Mtype;
  Mtype power = 0.37;
  Mtype scale = 0.83;
  Mtype shift = -2.4;
  this->TestBackward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerGradientShiftZero) {
  typedef typename TypeParam::Mtype Mtype;
  Mtype power = 0.37;
  Mtype scale = 0.83;
  Mtype shift = 0.0;
  this->TestBackward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerZero) {
  typedef typename TypeParam::Mtype Mtype;
  Mtype power = 0.0;
  Mtype scale = 0.83;
  Mtype shift = -2.4;
  this->TestForward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerZeroGradient) {
  typedef typename TypeParam::Mtype Mtype;
  Mtype power = 0.0;
  Mtype scale = 0.83;
  Mtype shift = -2.4;
  this->TestBackward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerOne) {
  typedef typename TypeParam::Mtype Mtype;
  Mtype power = 1.0;
  Mtype scale = 0.83;
  Mtype shift = -2.4;
  this->TestForward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerOneGradient) {
  typedef typename TypeParam::Mtype Mtype;
  Mtype power = 1.0;
  Mtype scale = 0.83;
  Mtype shift = -2.4;
  this->TestBackward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerTwo) {
  typedef typename TypeParam::Mtype Mtype;
  Mtype power = 2.0;
  Mtype scale = 0.34;
  Mtype shift = -2.4;
  this->TestForward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerTwoGradient) {
  typedef typename TypeParam::Mtype Mtype;
  Mtype power = 2.0;
  Mtype scale = 0.83;
  Mtype shift = -2.4;
  this->TestBackward(power, scale, shift);
}

TYPED_TEST(PowerLayerTest, TestPowerTwoScaleHalfGradient) {
  typedef typename TypeParam::Mtype Mtype;
  Mtype power = 2.0;
  Mtype scale = 0.5;
  Mtype shift = -2.4;
  this->TestBackward(power, scale, shift);
}

}  // namespace caffe
