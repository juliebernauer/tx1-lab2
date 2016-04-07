#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class EmbedLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
 protected:
  EmbedLayerTest()
      : blob_bottom_(new Blob<Dtype,Mtype>(4, 1, 1, 1)),
        blob_top_(new Blob<Dtype,Mtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype,Mtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~EmbedLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype,Mtype>* const blob_bottom_;
  Blob<Dtype,Mtype>* const blob_top_;
  vector<Blob<Dtype,Mtype>*> blob_bottom_vec_;
  vector<Blob<Dtype,Mtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(EmbedLayerTest, TestDtypesAndDevices);

TYPED_TEST(EmbedLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  EmbedParameter* embed_param = layer_param.mutable_embed_param();
  embed_param->set_num_output(10);
  embed_param->set_input_dim(5);
  shared_ptr<EmbedLayer<Dtype,Mtype> > layer(new EmbedLayer<Dtype,Mtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 5);
  EXPECT_EQ(this->blob_top_->shape(0), 4);
  EXPECT_EQ(this->blob_top_->shape(1), 1);
  EXPECT_EQ(this->blob_top_->shape(2), 1);
  EXPECT_EQ(this->blob_top_->shape(3), 1);
  EXPECT_EQ(this->blob_top_->shape(4), 10);
}

TYPED_TEST(EmbedLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  EmbedParameter* embed_param = layer_param.mutable_embed_param();
  const int kNumOutput = 10;
  const int kInputDim = 5;
  embed_param->set_num_output(kNumOutput);
  embed_param->set_input_dim(kInputDim);
  embed_param->mutable_weight_filler()->set_type("uniform");
  embed_param->mutable_weight_filler()->set_min(-10);
  embed_param->mutable_weight_filler()->set_max(10);
  embed_param->set_bias_term(false);
  shared_ptr<EmbedLayer<Dtype,Mtype> > layer(new EmbedLayer<Dtype,Mtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(1, layer->blobs().size());
  vector<int> weight_shape(2);
  weight_shape[0] = kInputDim;
  weight_shape[1] = kNumOutput;
  ASSERT_TRUE(weight_shape == layer->blobs()[0]->shape());
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    this->blob_bottom_->mutable_cpu_data()[i] = Get<Dtype>(caffe_rng_rand() % kInputDim);
  }
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<int> weight_offset(2, 0);
  vector<int> top_offset(5, 0);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    weight_offset[0] = Get<int>(this->blob_bottom_->cpu_data()[i]);
    weight_offset[1] = 0;
    top_offset[0] = i;
    top_offset[4] = 0;
    for (int j = 0; j < kNumOutput; ++j) {
      EXPECT_EQ(layer->blobs()[0]->data_at(weight_offset),
                this->blob_top_->data_at(top_offset));
      ++top_offset[4];
      ++weight_offset[1];
    }
  }
}

TYPED_TEST(EmbedLayerTest, TestForwardWithBias) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  EmbedParameter* embed_param = layer_param.mutable_embed_param();
  const int kNumOutput = 10;
  const int kInputDim = 5;
  embed_param->set_num_output(kNumOutput);
  embed_param->set_input_dim(kInputDim);
  embed_param->mutable_weight_filler()->set_type("uniform");
  embed_param->mutable_weight_filler()->set_min(-10);
  embed_param->mutable_weight_filler()->set_max(10);
  embed_param->mutable_bias_filler()->CopyFrom(embed_param->weight_filler());
  embed_param->set_bias_term(true);
  shared_ptr<EmbedLayer<Dtype,Mtype> > layer(new EmbedLayer<Dtype,Mtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(2, layer->blobs().size());
  vector<int> weight_shape(2);
  weight_shape[0] = kInputDim;
  weight_shape[1] = kNumOutput;
  ASSERT_TRUE(weight_shape == layer->blobs()[0]->shape());
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    this->blob_bottom_->mutable_cpu_data()[i] = Get<Dtype>(caffe_rng_rand() % kInputDim);
  }
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<int> bias_offset(1, 0);
  vector<int> weight_offset(2, 0);
  vector<int> top_offset(5, 0);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    weight_offset[0] = Get<int>(this->blob_bottom_->cpu_data()[i]);
    weight_offset[1] = 0;
    top_offset[0] = i;
    top_offset[4] = 0;
    bias_offset[0] = 0;
    for (int j = 0; j < kNumOutput; ++j) {
      EXPECT_EQ(layer->blobs()[0]->data_at(weight_offset) +
                layer->blobs()[1]->data_at(bias_offset),
                this->blob_top_->data_at(top_offset));
      ++top_offset[4];
      ++weight_offset[1];
      ++bias_offset[0];
    }
  }
}

TYPED_TEST(EmbedLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  EmbedParameter* embed_param = layer_param.mutable_embed_param();
  embed_param->set_num_output(10);
  embed_param->set_input_dim(5);
  embed_param->set_bias_term(false);
  embed_param->mutable_weight_filler()->set_type("uniform");
  embed_param->mutable_weight_filler()->set_min(-10);
  embed_param->mutable_weight_filler()->set_max(10);
  EmbedLayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3));
  this->blob_bottom_->mutable_cpu_data()[0] = Get<Dtype>(4);
  this->blob_bottom_->mutable_cpu_data()[1] = Get<Dtype>(2);
  this->blob_bottom_->mutable_cpu_data()[2] = Get<Dtype>(2);
  this->blob_bottom_->mutable_cpu_data()[3] = Get<Dtype>(3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, -2);
}

TYPED_TEST(EmbedLayerTest, TestGradientWithBias) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  EmbedParameter* embed_param = layer_param.mutable_embed_param();
  embed_param->set_num_output(10);
  embed_param->set_input_dim(5);
  embed_param->set_bias_term(true);
  embed_param->mutable_weight_filler()->set_type("uniform");
  embed_param->mutable_weight_filler()->set_min(-10);
  embed_param->mutable_weight_filler()->set_max(10);
  embed_param->mutable_bias_filler()->CopyFrom(embed_param->weight_filler());
  EmbedLayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3));
  this->blob_bottom_->mutable_cpu_data()[0] = Get<Dtype>(4);
  this->blob_bottom_->mutable_cpu_data()[1] = Get<Dtype>(2);
  this->blob_bottom_->mutable_cpu_data()[2] = Get<Dtype>(2);
  this->blob_bottom_->mutable_cpu_data()[3] = Get<Dtype>(3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, -2);
}

}  // namespace caffe
