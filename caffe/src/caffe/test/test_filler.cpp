#include <cstring>

#include "gtest/gtest.h"

#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class ConstantFillerTest : public ::testing::Test {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
 protected:
  ConstantFillerTest()
      : blob_(new Blob<Dtype,Mtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_param_.set_value(10.);
    filler_.reset(new ConstantFiller<Dtype,Mtype>(filler_param_));
    filler_->Fill(blob_);
  }
  virtual ~ConstantFillerTest() { delete blob_; }
  Blob<Dtype,Mtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<ConstantFiller<Dtype,Mtype> > filler_;
};

TYPED_TEST_CASE(ConstantFillerTest, TestDtypes);

TYPED_TEST(ConstantFillerTest, TestFill) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  EXPECT_TRUE(this->blob_);
  const int count = this->blob_->count();
  const Dtype* data = this->blob_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(Get<Mtype>(data[i]), Get<Mtype>(this->filler_param_.value()));
  }
}


template <typename TypeParam>
class UniformFillerTest : public ::testing::Test {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
 protected:
  UniformFillerTest()
      : blob_(new Blob<Dtype,Mtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_param_.set_min(1.);
    filler_param_.set_max(2.);
    filler_.reset(new UniformFiller<Dtype,Mtype>(filler_param_));
    filler_->Fill(blob_);
  }
  virtual ~UniformFillerTest() { delete blob_; }
  Blob<Dtype,Mtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<UniformFiller<Dtype,Mtype> > filler_;
};

TYPED_TEST_CASE(UniformFillerTest, TestDtypes);

TYPED_TEST(UniformFillerTest, TestFill) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  EXPECT_TRUE(this->blob_);
  const int count = this->blob_->count();
  const Dtype* data = this->blob_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(Get<Mtype>(data[i]), Get<Mtype>(this->filler_param_.min()));
    EXPECT_LE(Get<Mtype>(data[i]), Get<Mtype>(this->filler_param_.max()));
  }
}

template <typename TypeParam>
class PositiveUnitballFillerTest : public ::testing::Test {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
 protected:
  PositiveUnitballFillerTest()
      : blob_(new Blob<Dtype,Mtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_.reset(new PositiveUnitballFiller<Dtype,Mtype>(filler_param_));
    filler_->Fill(blob_);
  }
  virtual ~PositiveUnitballFillerTest() { delete blob_; }
  Blob<Dtype,Mtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<PositiveUnitballFiller<Dtype,Mtype> > filler_;
};

TYPED_TEST_CASE(PositiveUnitballFillerTest, TestDtypes);

TYPED_TEST(PositiveUnitballFillerTest, TestFill) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  EXPECT_TRUE(this->blob_);
  const int num = this->blob_->num();
  const int count = this->blob_->count();
  const int dim = count / num;
  const Dtype* data = this->blob_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(Get<Mtype>(data[i]), 0);
    EXPECT_LE(Get<Mtype>(data[i]), 1);
  }
  for (int i = 0; i < num; ++i) {
    Mtype sum = 0;
    for (int j = 0; j < dim; ++j) {
      sum += Get<Mtype>(data[i * dim + j]);
    }
    EXPECT_GE(sum, 0.999);
    EXPECT_LE(sum, 1.001);
  }
}

template <typename TypeParam>
class GaussianFillerTest : public ::testing::Test {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
 protected:
  GaussianFillerTest()
      : blob_(new Blob<Dtype,Mtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_param_.set_mean(10.);
    filler_param_.set_std(0.1);
    filler_.reset(new GaussianFiller<Dtype,Mtype>(filler_param_));
    filler_->Fill(blob_);
  }
  virtual ~GaussianFillerTest() { delete blob_; }
  Blob<Dtype,Mtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<GaussianFiller<Dtype,Mtype> > filler_;
};

TYPED_TEST_CASE(GaussianFillerTest, TestDtypes);

TYPED_TEST(GaussianFillerTest, TestFill) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  EXPECT_TRUE(this->blob_);
  const int count = this->blob_->count();
  const Dtype* data = this->blob_->cpu_data();
  Mtype mean = 0.;
  Mtype var = 0.;
  for (int i = 0; i < count; ++i) {
    mean += Get<Mtype>(data[i]);
    var += (Get<Mtype>(data[i]) - this->filler_param_.mean()) *
        (Get<Mtype>(data[i]) - this->filler_param_.mean());
  }
  mean /= count;
  var /= count;
  // Very loose test.
  EXPECT_GE(mean, this->filler_param_.mean() - this->filler_param_.std() * 5);
  EXPECT_LE(mean, this->filler_param_.mean() + this->filler_param_.std() * 5);
  Mtype target_var = this->filler_param_.std() * this->filler_param_.std();
  EXPECT_GE(var, target_var / 5.);
  EXPECT_LE(var, target_var * 5.);
}

template <typename TypeParam>
class XavierFillerTest : public ::testing::Test {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
 protected:
  XavierFillerTest()
      : blob_(new Blob<Dtype,Mtype>(1000, 2, 4, 5)),
        filler_param_() {
  }
  virtual void test_params(FillerParameter_VarianceNorm variance_norm,
      Dtype n) {
    this->filler_param_.set_variance_norm(variance_norm);
    this->filler_.reset(new XavierFiller<Dtype,Mtype>(this->filler_param_));
    this->filler_->Fill(blob_);
    EXPECT_TRUE(this->blob_);
    const int count = this->blob_->count();
    const Dtype* data = this->blob_->cpu_data();
    Dtype mean = Get<Dtype>(0.);
    Dtype ex2 = Get<Dtype>(0.);
    for (int i = 0; i < count; ++i) {
      mean += data[i];
      ex2 += data[i] * data[i];
    }
    mean /= count;
    ex2 /= count;
    Dtype std = Get<Dtype>(sqrt(Get<float>(ex2 - mean*mean)));
    Dtype target_std = Get<Dtype>(sqrt(2.0 / Get<float>(n)));
    EXPECT_NEAR(Get<float>(mean), 0.0, 0.1);
    EXPECT_NEAR(Get<float>(std), Get<float>(target_std), 0.1);
  }
  virtual ~XavierFillerTest() { delete blob_; }
  Blob<Dtype,Mtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<XavierFiller<Dtype,Mtype> > filler_;
};

TYPED_TEST_CASE(XavierFillerTest, TestDtypes);

TYPED_TEST(XavierFillerTest, TestFillFanIn) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype n = Get<Dtype>(2*4*5);
  this->test_params(FillerParameter_VarianceNorm_FAN_IN, n);
}
TYPED_TEST(XavierFillerTest, TestFillFanOut) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype n = Get<Dtype>(1000*4*5);
  this->test_params(FillerParameter_VarianceNorm_FAN_OUT, n);
}
TYPED_TEST(XavierFillerTest, TestFillAverage) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype n = Get<Dtype>((2*4*5 + 1000*4*5) / 2.0);
  this->test_params(FillerParameter_VarianceNorm_AVERAGE, n);
}

template <typename TypeParam>
class MSRAFillerTest : public ::testing::Test {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
 protected:
  MSRAFillerTest()
      : blob_(new Blob<Dtype,Mtype>(1000, 2, 4, 5)),
        filler_param_() {
  }
  virtual void test_params(FillerParameter_VarianceNorm variance_norm,
      Dtype n) {
    this->filler_param_.set_variance_norm(variance_norm);
    this->filler_.reset(new MSRAFiller<Dtype,Mtype>(this->filler_param_));
    this->filler_->Fill(blob_);
    EXPECT_TRUE(this->blob_);
    const int count = this->blob_->count();
    const Dtype* data = this->blob_->cpu_data();
    Dtype mean = Get<Dtype>(0.);
    Dtype ex2 = Get<Dtype>(0.);
    for (int i = 0; i < count; ++i) {
      mean += data[i];
      ex2 += data[i] * data[i];
    }
    mean /= count;
    ex2 /= count;
    Dtype std = Get<Dtype>(sqrt(Get<float>(ex2 - mean*mean)));
    Dtype target_std = Get<Dtype>(sqrt(2.0F / Get<float>(n)));
    EXPECT_NEAR(Get<float>(mean), 0.0, 0.1);
    EXPECT_NEAR(Get<float>(std), Get<float>(target_std), 0.1);
  }
  virtual ~MSRAFillerTest() { delete blob_; }
  Blob<Dtype,Mtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<MSRAFiller<Dtype,Mtype> > filler_;
};

TYPED_TEST_CASE(MSRAFillerTest, TestDtypes);

TYPED_TEST(MSRAFillerTest, TestFillFanIn) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype n = Get<Dtype>(2*4*5);
  this->test_params(FillerParameter_VarianceNorm_FAN_IN, n);
}
TYPED_TEST(MSRAFillerTest, TestFillFanOut) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype n = Get<Dtype>(1000*4*5);
  this->test_params(FillerParameter_VarianceNorm_FAN_OUT, n);
}
TYPED_TEST(MSRAFillerTest, TestFillAverage) {
  typedef typename TypeParam::Dtype Dtype;
  Dtype n = Get<Dtype>((2*4*5 + 1000*4*5) / 2.0);
  this->test_params(FillerParameter_VarianceNorm_AVERAGE, n);
}

}  // namespace caffe
