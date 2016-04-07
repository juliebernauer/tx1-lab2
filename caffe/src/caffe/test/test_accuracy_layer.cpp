#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class AccuracyLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
 protected:
  AccuracyLayerTest()
      : blob_bottom_data_(new Blob<Dtype,Mtype>()),
        blob_bottom_label_(new Blob<Dtype,Mtype>()),
        blob_top_(new Blob<Dtype,Mtype>()),
        blob_top_per_class_(new Blob<Dtype,Mtype>()),
        top_k_(3) {
    vector<int> shape(2);
    shape[0] = 100;
    shape[1] = 10;
    blob_bottom_data_->Reshape(shape);
    shape.resize(1);
    blob_bottom_label_->Reshape(shape);
    FillBottoms();

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_per_class_vec_.push_back(blob_top_);
    blob_top_per_class_vec_.push_back(blob_top_per_class_);
  }

  virtual void FillBottoms() {
    // fill the probability values
    FillerParameter filler_param;
    GaussianFiller<Dtype,Mtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);

    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    shared_ptr<Caffe::RNG> rng(new Caffe::RNG(prefetch_rng_seed));
    caffe::rng_t* prefetch_rng =
          static_cast<caffe::rng_t*>(rng->generator());
    Dtype* label_data = blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      label_data[i] = Get<Dtype>((*prefetch_rng)() % 10);
    }
  }

  virtual ~AccuracyLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_;
    delete blob_top_per_class_;
  }
  Blob<Dtype,Mtype>* const blob_bottom_data_;
  Blob<Dtype,Mtype>* const blob_bottom_label_;
  Blob<Dtype,Mtype>* const blob_top_;
  Blob<Dtype,Mtype>* const blob_top_per_class_;
  vector<Blob<Dtype,Mtype>*> blob_bottom_vec_;
  vector<Blob<Dtype,Mtype>*> blob_top_vec_;
  vector<Blob<Dtype,Mtype>*> blob_top_per_class_vec_;
  int top_k_;
};

TYPED_TEST_CASE(AccuracyLayerTest, TestDtypes);

TYPED_TEST(AccuracyLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  AccuracyLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(AccuracyLayerTest, TestSetupTopK) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  AccuracyParameter* accuracy_param =
      layer_param.mutable_accuracy_param();
  accuracy_param->set_top_k(5);
  AccuracyLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(AccuracyLayerTest, TestSetupOutputPerClass) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  AccuracyLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_per_class_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_per_class_->num(), 10);
  EXPECT_EQ(this->blob_top_per_class_->channels(), 1);
  EXPECT_EQ(this->blob_top_per_class_->height(), 1);
  EXPECT_EQ(this->blob_top_per_class_->width(), 1);
}

TYPED_TEST(AccuracyLayerTest, TestForwardCPU) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  AccuracyLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  Mtype max_value;
  int max_id;
  int num_correct_labels = 0;
  for (int i = 0; i < 100; ++i) {
    max_value = Get<Mtype>(-FLT_MAX);
    max_id = 0;
    for (int j = 0; j < 10; ++j) {
      if (Get<Mtype>(this->blob_bottom_data_->data_at(i, j, 0, 0)) > max_value) {
        max_value = Get<Mtype>(this->blob_bottom_data_->data_at(i, j, 0, 0));
        max_id = j;
      }
    }
    if (max_id == Get<int>(this->blob_bottom_label_->data_at(i, 0, 0, 0))) {
      ++num_correct_labels;
    }
  }
  EXPECT_NEAR(Get<Mtype>(this->blob_top_->data_at(0, 0, 0, 0)),
              num_correct_labels / 100.0, 1e-4);
}

TYPED_TEST(AccuracyLayerTest, TestForwardWithSpatialAxes) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  this->blob_bottom_data_->Reshape(2, 10, 4, 5);
  vector<int> label_shape(3);
  label_shape[0] = 2; label_shape[1] = 4; label_shape[2] = 5;
  this->blob_bottom_label_->Reshape(label_shape);
  this->FillBottoms();
  LayerParameter layer_param;
  layer_param.mutable_accuracy_param()->set_axis(1);
  AccuracyLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  Mtype max_value;
  const int num_labels = this->blob_bottom_label_->count();
  int max_id;
  int num_correct_labels = 0;
  vector<int> label_offset(3);
  for (int n = 0; n < this->blob_bottom_data_->num(); ++n) {
    for (int h = 0; h < this->blob_bottom_data_->height(); ++h) {
      for (int w = 0; w < this->blob_bottom_data_->width(); ++w) {
        max_value = Get<Mtype>(-FLT_MAX);
        max_id = 0;
        for (int c = 0; c < this->blob_bottom_data_->channels(); ++c) {
          const Mtype pred_value =
              Get<Mtype>(this->blob_bottom_data_->data_at(n, c, h, w));
          if (pred_value > max_value) {
            max_value = pred_value;
            max_id = c;
          }
        }
        label_offset[0] = n; label_offset[1] = h; label_offset[2] = w;
        const int correct_label =
            Get<int>(this->blob_bottom_label_->data_at(label_offset));
        if (max_id == correct_label) {
          ++num_correct_labels;
        }
      }
    }
  }
  EXPECT_NEAR(Get<Mtype>(this->blob_top_->data_at(0, 0, 0, 0)),
              num_correct_labels / Get<Mtype>(num_labels), 1e-4);
}

TYPED_TEST(AccuracyLayerTest, TestForwardIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  const Mtype kIgnoreLabelValue = -1.;
  layer_param.mutable_accuracy_param()->set_ignore_label(kIgnoreLabelValue);
  AccuracyLayer<Dtype,Mtype> layer(layer_param);
  // Manually set some labels to the ignore label value (-1).
  this->blob_bottom_label_->mutable_cpu_data()[2] = Get<Dtype>(kIgnoreLabelValue);
  this->blob_bottom_label_->mutable_cpu_data()[5] = Get<Dtype>(kIgnoreLabelValue);
  this->blob_bottom_label_->mutable_cpu_data()[32] = Get<Dtype>(kIgnoreLabelValue);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  Mtype max_value;
  int max_id;
  int num_correct_labels = 0;
  int count = 0;
  for (int i = 0; i < 100; ++i) {
    if (kIgnoreLabelValue == Get<Mtype>(this->blob_bottom_label_->data_at(i, 0, 0, 0))) {
      continue;
    }
    ++count;
    max_value = Get<Mtype>(-FLT_MAX);
    max_id = 0;
    for (int j = 0; j < 10; ++j) {
      if (Get<Mtype>(this->blob_bottom_data_->data_at(i, j, 0, 0)) > max_value) {
        max_value = Get<Mtype>(this->blob_bottom_data_->data_at(i, j, 0, 0));
        max_id = j;
      }
    }
    if (max_id == Get<Mtype>(this->blob_bottom_label_->data_at(i, 0, 0, 0))) {
      ++num_correct_labels;
    }
  }
  EXPECT_EQ(count, 97);  // We set 3 out of 100 labels to kIgnoreLabelValue.
  EXPECT_NEAR(Get<Mtype>(this->blob_top_->data_at(0, 0, 0, 0)),
              num_correct_labels / Get<Mtype>(count), 1e-4);
}

TYPED_TEST(AccuracyLayerTest, TestForwardCPUTopK) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  AccuracyParameter* accuracy_param = layer_param.mutable_accuracy_param();
  accuracy_param->set_top_k(this->top_k_);
  AccuracyLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  Mtype current_value;
  int current_rank;
  int num_correct_labels = 0;
  for (int i = 0; i < 100; ++i) {
    for (int j = 0; j < 10; ++j) {
      current_value = Get<Mtype>(this->blob_bottom_data_->data_at(i, j, 0, 0));
      current_rank = 0;
      for (int k = 0; k < 10; ++k) {
        if (Get<Mtype>(this->blob_bottom_data_->data_at(i, k, 0, 0)) > current_value) {
          ++current_rank;
        }
      }
      if (current_rank < this->top_k_ &&
          j == Get<int>(this->blob_bottom_label_->data_at(i, 0, 0, 0))) {
        ++num_correct_labels;
      }
    }
  }

  EXPECT_NEAR(Get<Mtype>(this->blob_top_->data_at(0, 0, 0, 0)),
              num_correct_labels / 100.0, 1e-4);
}

TYPED_TEST(AccuracyLayerTest, TestForwardCPUPerClass) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  AccuracyLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_per_class_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_per_class_vec_);

  Dtype max_value;
  int max_id;
  int num_correct_labels = 0;
  const int num_class = this->blob_top_per_class_->num();
  vector<int> correct_per_class(num_class, 0);
  vector<int> num_per_class(num_class, 0);
  for (int i = 0; i < 100; ++i) {
    max_value = Get<Dtype>(- maxDtype<Dtype>());
    max_id = 0;
    for (int j = 0; j < 10; ++j) {
      if (this->blob_bottom_data_->data_at(i, j, 0, 0) > max_value) {
        max_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
        max_id = j;
      }
    }
    const int id = Get<int>(this->blob_bottom_label_->data_at(i, 0, 0, 0));
    ++num_per_class[id];
    if (max_id == id) {
      ++num_correct_labels;
      ++correct_per_class[max_id];
    }
  }
  EXPECT_NEAR(Get<Mtype>(this->blob_top_->data_at(0, 0, 0, 0)),
              num_correct_labels / 100.0, 1e-4);
  for (int i = 0; i < num_class; ++i) {
    float accuracy_per_class = num_per_class[i] > 0 ?
    		correct_per_class[i] / num_per_class[i] : 0.;
    EXPECT_NEAR(Get<Mtype>(this->blob_top_per_class_->data_at(i, 0, 0, 0)),
    		Get<Mtype>(accuracy_per_class), 1e-4);
  }
}


TYPED_TEST(AccuracyLayerTest, TestForwardCPUPerClassWithIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  const int kIgnoreLabelValue = -1;
  layer_param.mutable_accuracy_param()->set_ignore_label(kIgnoreLabelValue);
  AccuracyLayer<Dtype,Mtype> layer(layer_param);
  // Manually set some labels to the ignore label value (-1).
  this->blob_bottom_label_->mutable_cpu_data()[2] = Get<Dtype>(kIgnoreLabelValue);
  this->blob_bottom_label_->mutable_cpu_data()[5] = Get<Dtype>(kIgnoreLabelValue);
  this->blob_bottom_label_->mutable_cpu_data()[32] = Get<Dtype>(kIgnoreLabelValue);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_per_class_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_per_class_vec_);

  Dtype max_value;
  int max_id;
  int num_correct_labels = 0;
  const int num_class = this->blob_top_per_class_->num();
  vector<int> correct_per_class(num_class, 0);
  vector<int> num_per_class(num_class, 0);
  int count = 0;
  for (int i = 0; i < 100; ++i) {
    if (Get<Dtype>(kIgnoreLabelValue) == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
      continue;
    }
    ++count;
    max_value = Get<Dtype>(- maxDtype<Dtype>());
    max_id = 0;
    for (int j = 0; j < 10; ++j) {
      if (this->blob_bottom_data_->data_at(i, j, 0, 0) > max_value) {
        max_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
        max_id = j;
      }
    }
    const int id = Get<int>(this->blob_bottom_label_->data_at(i, 0, 0, 0));
    ++num_per_class[id];
    if (max_id == id) {
      ++num_correct_labels;
      ++correct_per_class[max_id];
    }
  }
  EXPECT_EQ(count, 97);
  EXPECT_NEAR(Get<Mtype>(this->blob_top_->data_at(0, 0, 0, 0)),
              num_correct_labels / Get<Mtype>(count), 1e-4);
  for (int i = 0; i < 10; ++i) {
    float accuracy_per_class = num_per_class[i] > 0 ?
    		correct_per_class[i] / num_per_class[i] : 0.;
    EXPECT_NEAR(Get<Mtype>(this->blob_top_per_class_->data_at(i, 0, 0, 0)),
    		Get<Mtype>(accuracy_per_class), 1e-4);
  }
}

}  // namespace caffe
