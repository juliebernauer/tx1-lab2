// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#ifndef CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
#define CAFFE_TEST_TEST_CAFFE_MAIN_HPP_

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include "caffe/common.hpp"

using std::cout;
using std::endl;

#ifdef CMAKE_BUILD
  #include "caffe_config.h"
#else
  #define CUDA_TEST_DEVICE -1
  #define CMAKE_SOURCE_DIR "src/"
  #define EXAMPLES_SOURCE_DIR "examples/"
  #define CMAKE_EXT ""
#endif

int main(int argc, char** argv);

namespace caffe {

template <typename S, typename M>
struct MultiPrecision
{
  typedef S Dtype;
  typedef M Mtype;
};

template <typename TypeParam>
class MultiDeviceTest : public ::testing::Test {
 public:
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
 protected:
  MultiDeviceTest() {
    Caffe::set_mode(TypeParam::device);
  }
  virtual ~MultiDeviceTest() {}
};

#ifdef CPU_ONLY
typedef ::testing::Types<MultiPrecision<float,float>,
                         MultiPrecision<double,double> > TestDtypes;
#else
typedef ::testing::Types<MultiPrecision<float,float>,
                         MultiPrecision<double,double>,
                         MultiPrecision<float16,CAFFE_FP16_MTYPE> > TestDtypes;
#endif

template <typename TypeParam>
struct CPUDevice {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  static const Caffe::Brew device = Caffe::CPU;
};

template <typename TypeParam>
class CPUDeviceTest : public MultiDeviceTest<CPUDevice<TypeParam> > {
};

#ifdef CPU_ONLY
typedef ::testing::Types<CPUDevice<MultiPrecision<float,float> >,
                         CPUDevice<MultiPrecision<double,double> > > TestDtypesAndDevices;
#else

template <typename TypeParam>
struct GPUDevice {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  static const Caffe::Brew device = Caffe::GPU;
};

template <typename TypeParam>
class GPUDeviceTest : public MultiDeviceTest<GPUDevice<TypeParam> > {
};

typedef ::testing::Types<CPUDevice<MultiPrecision<float,float> >,
                         CPUDevice<MultiPrecision<double,double> >,
                         CPUDevice<MultiPrecision<float16,CAFFE_FP16_MTYPE> >,
                         GPUDevice<MultiPrecision<float,float> >,
                         GPUDevice<MultiPrecision<double, double> >,
                         GPUDevice<MultiPrecision<float16,CAFFE_FP16_MTYPE> > > TestDtypesAndDevices;
#endif

}  // namespace caffe

#endif  // CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
