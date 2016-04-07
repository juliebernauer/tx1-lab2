#ifndef CPU_ONLY  // CPU-GPU test

#include <cstring>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename TypeParam>
class GemmTest : public ::testing::Test {};

TYPED_TEST_CASE(GemmTest, TestDtypes);

TYPED_TEST(GemmTest, TestGemmCPUGPU) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  Blob<Dtype,Mtype> A(1, 1, 2, 3);
  Blob<Dtype,Mtype> B(1, 1, 3, 4);
  Blob<Dtype,Mtype> C(1, 1, 2, 4);
  Dtype data[12] = {Get<Dtype>(1), Get<Dtype>(2), Get<Dtype>(3), Get<Dtype>(4),
                    Get<Dtype>(5), Get<Dtype>(6), Get<Dtype>(7), Get<Dtype>(8),
                    Get<Dtype>(9), Get<Dtype>(10), Get<Dtype>(11), Get<Dtype>(12)};
  Dtype A_reshape_data[6] = {Get<Dtype>(1), Get<Dtype>(4), Get<Dtype>(2), Get<Dtype>(5),
                             Get<Dtype>(3), Get<Dtype>(6)};
  Dtype B_reshape_data[12] = {Get<Dtype>(1), Get<Dtype>(5), Get<Dtype>(9), Get<Dtype>(2),
                              Get<Dtype>(6), Get<Dtype>(10), Get<Dtype>(3), Get<Dtype>(7),
                              Get<Dtype>(11), Get<Dtype>(4), Get<Dtype>(8), Get<Dtype>(12)};
  Dtype result[8] = {Get<Dtype>(38), Get<Dtype>(44), Get<Dtype>(50), Get<Dtype>(56),
                     Get<Dtype>(83), Get<Dtype>(98), Get<Dtype>(113), Get<Dtype>(128)};
  caffe_copy<Dtype,Mtype>(6, data, A.mutable_cpu_data());
  caffe_copy<Dtype,Mtype>(12, data, B.mutable_cpu_data());

  if (sizeof(Dtype) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    // [1, 2, 3; 4 5 6] * [1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12];
    caffe_cpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(Get<Mtype>(result[i]), Get<Mtype>(C.cpu_data()[i]));
    }
    caffe_gpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(Get<Mtype>(result[i]), Get<Mtype>(C.cpu_data()[i]));
    }

    // Test when we have a transposed A
    A.Reshape(1, 1, 3, 2);
    caffe_copy<Dtype,Mtype>(6, A_reshape_data, A.mutable_cpu_data());
    caffe_cpu_gemm<Dtype,Mtype>(CblasTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(Get<Mtype>(result[i]), Get<Mtype>(C.cpu_data()[i]));
    }
    caffe_gpu_gemm<Dtype,Mtype>(CblasTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(Get<Mtype>(result[i]), Get<Mtype>(C.cpu_data()[i]));
    }

    // Test when we have a transposed A and a transposed B too
    B.Reshape(1, 1, 4, 3);
    caffe_copy<Dtype,Mtype>(12, B_reshape_data, B.mutable_cpu_data());
    caffe_cpu_gemm<Dtype,Mtype>(CblasTrans, CblasTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(Get<Mtype>(result[i]), Get<Mtype>(C.cpu_data()[i]));
    }
    caffe_gpu_gemm<Dtype,Mtype>(CblasTrans, CblasTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(Get<Mtype>(result[i]), Get<Mtype>(C.cpu_data()[i]));
    }

    // Test when we have a transposed B
    A.Reshape(1, 1, 2, 3);
    caffe_copy<Dtype,Mtype>(6, data, A.mutable_cpu_data());
    caffe_cpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(Get<Mtype>(result[i]), Get<Mtype>(C.cpu_data()[i]));
    }
    caffe_gpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(Get<Mtype>(result[i]), Get<Mtype>(C.cpu_data()[i]));
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GemmTest, TestGemmCPUGPUbeta1) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  Blob<Dtype,Mtype> A(1, 1, 3, 2);
  Blob<Dtype,Mtype> B(1, 1, 2, 1);
  Blob<Dtype,Mtype> C(1, 1, 3, 1);
  Dtype data[6] = {Get<Dtype>(1), Get<Dtype>(2),
                   Get<Dtype>(3), Get<Dtype>(4),
                   Get<Dtype>(5), Get<Dtype>(6)};
  Dtype result[3] = {Get<Dtype>(5), Get<Dtype>(11), Get<Dtype>(17)};
  caffe_copy<Dtype,Mtype>(6, data, A.mutable_cpu_data());
  caffe_copy<Dtype,Mtype>(2, data, B.mutable_cpu_data());

  if (sizeof(Dtype) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    caffe_copy<Dtype,Mtype>(3, result, C.mutable_cpu_data());
    caffe_cpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasNoTrans, 3, 1, 2, 1.,
        A.cpu_data(), B.cpu_data(), 1., C.mutable_cpu_data());
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(Get<Mtype>(result[i]) * Get<Mtype>(2.), Get<Mtype>(C.cpu_data()[i]));
    }
    caffe_copy<Dtype,Mtype>(3, result, C.mutable_cpu_data());
    caffe_gpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasNoTrans, 3, 1, 2, 1.,
        A.gpu_data(), B.gpu_data(), 1., C.mutable_gpu_data());
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(Get<Mtype>(result[i]) * Get<Mtype>(2.), Get<Mtype>(C.cpu_data()[i]));
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GemmTest, TestGemvCPUGPU) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  Blob<Dtype,Mtype> A(1, 1, 2, 3);
  Blob<Dtype,Mtype> x(1, 1, 1, 3);
  Blob<Dtype,Mtype> y(1, 1, 1, 2);
  Dtype data[6] = {Get<Dtype>(1), Get<Dtype>(2), Get<Dtype>(3),
                   Get<Dtype>(4), Get<Dtype>(5), Get<Dtype>(6)};
  Dtype result_2[2] = {Get<Dtype>(14), Get<Dtype>(32)};
  Dtype result_3[3] = {Get<Dtype>(9), Get<Dtype>(12), Get<Dtype>(15)};
  caffe_copy<Dtype,Mtype>(6, data, A.mutable_cpu_data());
  caffe_copy<Dtype,Mtype>(3, data, x.mutable_cpu_data());

  if (sizeof(Dtype) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    caffe_cpu_gemv<Dtype,Mtype>(CblasNoTrans, 2, 3, 1., A.cpu_data(),
        x.cpu_data(), 0., y.mutable_cpu_data());
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(Get<Mtype>(result_2[i]), Get<Mtype>(y.cpu_data()[i]));
    }
    caffe_gpu_gemv<Dtype,Mtype>(CblasNoTrans, 2, 3, 1., A.gpu_data(),
        x.gpu_data(), 0., y.mutable_gpu_data());
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(Get<Mtype>(result_2[i]), Get<Mtype>(y.cpu_data()[i]));
    }

    // Test transpose case
    caffe_copy<Dtype,Mtype>(2, data, y.mutable_cpu_data());
    caffe_cpu_gemv<Dtype,Mtype>(CblasTrans, 2, 3, 1., A.cpu_data(),
        y.cpu_data(), 0., x.mutable_cpu_data());
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(Get<Mtype>(result_3[i]), Get<Mtype>(x.cpu_data()[i]));
    }
    caffe_gpu_gemv<Dtype,Mtype>(CblasTrans, 2, 3, 1., A.gpu_data(),
        y.gpu_data(), 0., x.mutable_gpu_data());
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(Get<Mtype>(result_3[i]), Get<Mtype>(x.cpu_data()[i]));
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GemmTest, TestGemvCPUGPU2) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  Blob<Dtype,Mtype> A(1, 1, 3, 2);
  Blob<Dtype,Mtype> x(1, 1, 1, 2);
  Blob<Dtype,Mtype> y(1, 1, 1, 3);
  Dtype data[6] = {Get<Dtype>(1), Get<Dtype>(2),
                   Get<Dtype>(3), Get<Dtype>(4),
                   Get<Dtype>(5), Get<Dtype>(6)};
  Dtype result_3[3] = {Get<Dtype>(5), Get<Dtype>(11), Get<Dtype>(17)};
  Dtype result_2[2] = {Get<Dtype>(22), Get<Dtype>(28)};
  caffe_copy<Dtype,Mtype>(6, data, A.mutable_cpu_data());
  caffe_copy<Dtype,Mtype>(2, data, x.mutable_cpu_data());

  if (sizeof(Dtype) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    caffe_cpu_gemv<Dtype,Mtype>(CblasNoTrans, 3, 2, 1., A.cpu_data(),
        x.cpu_data(), 0., y.mutable_cpu_data());
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(Get<Mtype>(result_3[i]), Get<Mtype>(y.cpu_data()[i]));
    }
    caffe_gpu_gemv<Dtype,Mtype>(CblasNoTrans, 3, 2, 1., A.gpu_data(),
        x.gpu_data(), 0., y.mutable_gpu_data());
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(Get<Mtype>(result_3[i]), Get<Mtype>(y.cpu_data()[i]));
    }

    // Test transpose case
    caffe_copy<Dtype,Mtype>(3, data, y.mutable_cpu_data());
    caffe_cpu_gemv<Dtype,Mtype>(CblasTrans, 3, 2, 1., A.cpu_data(),
        y.cpu_data(), 0., x.mutable_cpu_data());
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(Get<Mtype>(result_2[i]), Get<Mtype>(x.cpu_data()[i]));
    }
    caffe_gpu_gemv<Dtype,Mtype>(CblasTrans, 3, 2, 1., A.gpu_data(),
        y.gpu_data(), 0., x.mutable_gpu_data());
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(Get<Mtype>(result_2[i]), Get<Mtype>(x.cpu_data()[i]));
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe

#endif  // CPU_ONLY
