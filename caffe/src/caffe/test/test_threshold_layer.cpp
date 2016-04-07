#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class ThresholdLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
 protected:
  ThresholdLayerTest()
      : blob_bottom_(new Blob<Dtype,Mtype>(2, 3, 6, 5)),
        blob_top_(new Blob<Dtype,Mtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype,Mtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ThresholdLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype,Mtype>* const blob_bottom_;
  Blob<Dtype,Mtype>* const blob_top_;
  vector<Blob<Dtype,Mtype>*> blob_bottom_vec_;
  vector<Blob<Dtype,Mtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ThresholdLayerTest, TestDtypesAndDevices);


TYPED_TEST(ThresholdLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  ThresholdLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(ThresholdLayerTest, Test) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  ThresholdLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Mtype threshold_ = layer_param.threshold_param().threshold();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(Get<Mtype>(top_data[i]), 0.);
    EXPECT_LE(Get<Mtype>(top_data[i]), 1.);
    if (Get<Mtype>(top_data[i]) == 0) {
      EXPECT_LE(Get<Mtype>(bottom_data[i]), threshold_);
    }
    if (Get<Mtype>(top_data[i]) == 1) {
      EXPECT_GT(Get<Mtype>(bottom_data[i]), threshold_);
    }
  }
}

TYPED_TEST(ThresholdLayerTest, Test2) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  ThresholdParameter* threshold_param =
    layer_param.mutable_threshold_param();
  threshold_param->set_threshold(0.5);
  ThresholdLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Mtype threshold_ = layer_param.threshold_param().threshold();
  EXPECT_FLOAT_EQ(threshold_, 0.5);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(Get<Mtype>(top_data[i]), 0.);
    EXPECT_LE(Get<Mtype>(top_data[i]), 1.);
    if (Get<Mtype>(top_data[i]) == 0) {
      EXPECT_LE(Get<Mtype>(bottom_data[i]), threshold_);
    }
    if (Get<Mtype>(top_data[i]) == 1) {
      EXPECT_GT(Get<Mtype>(bottom_data[i]), threshold_);
    }
  }
}

}  // namespace caffe
