#include <map>
#include <string>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class LayerFactoryTest : public MultiDeviceTest<TypeParam> {};

TYPED_TEST_CASE(LayerFactoryTest, TestDtypesAndDevices);

TYPED_TEST(LayerFactoryTest, TestCreateLayer) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  typename LayerRegistry<Dtype,Mtype>::CreatorRegistry& registry =
      LayerRegistry<Dtype,Mtype>::Registry();
  shared_ptr<Layer<Dtype,Mtype> > layer;
  for (typename LayerRegistry<Dtype,Mtype>::CreatorRegistry::iterator iter =
       registry.begin(); iter != registry.end(); ++iter) {
    // Special case: PythonLayer is checked by pytest
    if (iter->first == "Python") { continue; }
    LayerParameter layer_param;
    // Data layers expect a DB
    if (iter->first == "Data") {
#ifdef USE_LEVELDB
      string tmp;
      MakeTempDir(&tmp);
      boost::scoped_ptr<db::DB> db(db::GetDB(DataParameter_DB_LEVELDB));
      db->Open(tmp, db::NEW);
      db->Close();
      layer_param.mutable_data_param()->set_source(tmp);
#else
      continue;
#endif  // USE_LEVELDB
    }
    layer_param.set_type(iter->first);
    layer = LayerRegistry<Dtype,Mtype>::CreateLayer(layer_param);
    EXPECT_EQ(iter->first, layer->type());
  }
}

}  // namespace caffe
