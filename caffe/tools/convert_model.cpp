#include <stdint.h>
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/solver.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

DEFINE_string(model, "",
    "The model definition protocol buffer text file.");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  gflags::SetUsageMessage("Converts Caffe model file\n"
        "Usage:\n"
        "    convert_model [FLAGS] INPUT_MODEL OUTPUT_MODEL\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 3 || argc > 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_model");
    return 1;
  }

  map<std::string, shared_ptr<Blob<float16,CAFFE_FP16_MTYPE> > > blobs16map;
  map<std::string, shared_ptr<LayerParameter> > layerpar16map;
  int diff_size = 0;
  {
//    SolverParameter solver_param;
//    ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);
//    shared_ptr<caffe::Solver<float,float> >
//      solver(caffe::GetSolver<float,float>(solver_param));
//    shared_ptr<Net<float,float> > net = solver->net();

    shared_ptr<Net<float,float> > net(new Net<float,float>(FLAGS_model, caffe::TRAIN));
    const string trained_filename(argv[1]);
    net->CopyTrainedLayersFromBinaryProto(trained_filename);

    const vector<string>& blob_names = net->blob_names();
    for (vector<string>::const_iterator it = blob_names.begin();
        it != blob_names.end(); ++it) {
      const std::string& blob_name = *it;

      shared_ptr<Blob<float,float> > blob = net->blob_by_name(blob_name);
      BlobProto blob_proto, blob_proto16;
      blob->ToProto(&blob_proto, true);
      blob_proto16.mutable_shape()->CopyFrom(blob_proto.shape());

      const int data_size = blob_proto.data_size();
      blob_proto16.mutable_half_data()->Reserve(data_size);
      for (int i = 0; i < data_size; ++i) {
        blob_proto16.mutable_half_data()->Add(float16(blob_proto.data(i)).getx());
      }

      diff_size = blob_proto.diff_size();
      blob_proto16.mutable_half_diff()->Reserve(diff_size);
      for (int i = 0; i < diff_size; ++i) {
        blob_proto16.mutable_half_diff()->Add(float16(blob_proto.diff(i)).getx());
      }

      shared_ptr<Blob<float16,CAFFE_FP16_MTYPE> > blob16(new Blob<float16,CAFFE_FP16_MTYPE>);
      blob16->FromProto(blob_proto16);
      blobs16map[blob_name] = blob16;
    }

    std::cout << "Reading..." << std::endl;

    const vector<string>& layer_names = net->layer_names();
    for (vector<string>::const_iterator it = layer_names.begin();
        it != layer_names.end(); ++it) {
      const std::string& layer_name = *it;

      std::cout << "Layer: " << layer_name << std::endl;

      shared_ptr<Layer<float,float> > layer = net->layer_by_name(layer_name);
      const vector<shared_ptr<Blob<float,float> > >& blobs = layer->blobs();

      const LayerParameter& layer_param = layer->layer_param();
      shared_ptr<LayerParameter> layer_param16(new LayerParameter);
      layer_param16->CopyFrom(layer_param);
      layer_param16->clear_blobs();

      for (int i = 0; i < blobs.size(); ++i) {
        BlobProto blob_proto;
        blobs[i]->ToProto(&blob_proto, true);

        BlobProto* blob_proto16 = layer_param16->add_blobs();
        blob_proto16->mutable_shape()->CopyFrom(blob_proto.shape());
        const int data_size = blob_proto.data_size();
        blob_proto16->mutable_half_data()->Reserve(data_size);

        std::cout << "\tBlob " << i << ": " << data_size << std::endl;

        for (int j = 0; j < data_size; ++j) {
          blob_proto16->mutable_half_data()->Add(float16(blob_proto.data(j)).getx());
        }
      }
      layerpar16map[layer_name] = layer_param16;
    }
  }

  {
//    SolverParameter solver_param16;
//    ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param16);
//
//    shared_ptr<caffe::Solver<float16,CAFFE_FP16_MTYPE> >
//      solver16(caffe::GetSolver<float16,CAFFE_FP16_MTYPE>(solver_param16));
//    shared_ptr<Net<float16,CAFFE_FP16_MTYPE> > net16 = solver16->net();

    shared_ptr<Net<float16,CAFFE_FP16_MTYPE> >
    net16(new Net<float16,CAFFE_FP16_MTYPE>(FLAGS_model, caffe::TRAIN));
    const string trained_filename(argv[1]);
    net16->CopyTrainedLayersFromBinaryProto(trained_filename);

//  See for details:
//  void Net<Dtype,Mtype>::ToProto(NetParameter* param, bool write_diff) const
//
//    const vector<string>& blob_names = net16->blob_names();
//    for (vector<string>::const_iterator it = blob_names.begin();
//        it != blob_names.end(); ++it) {
//      const std::string& blob_name = *it;
//      net16->set_blob_by_name(blob_name, blobs16map[blob_name]);
//    }

    std::cout  << std::endl << "Writing..." << std::endl;

    NetParameter net_param16;
    net16->ToProto(&net_param16, diff_size > 0);
    net_param16.clear_layer();

    const vector<string>& layer_names = net16->layer_names();
    for (vector<string>::const_iterator it = layer_names.begin();
        it != layer_names.end(); ++it) {
      const std::string& layer_name = *it;
      std::cout << "Layer: " << layer_name << std::endl;
      net_param16.add_layer()->CopyFrom(*layerpar16map[layer_name]);
    }

//    int sz = net_param16.layer_size();
//    for (int k = 0 ; k < sz; ++k) {
//      const LayerParameter& lp = net_param16.layer(k);
//        if (lp.blobs_size() > 0) {
//          const BlobProto& b = lp.blobs(0);
//          int i0 = b.half_data(0);
//          int i1 = b.half_data(1);
//          std::cout << "*** " << i0 << " " << i1 << std::endl;
//        }
//    }

    WriteProtoToBinaryFile(net_param16, argv[2]);
  }


//  SolverParameter solver_param;
//  ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);
//
//  shared_ptr<caffe::Solver<float,float> >
//    solver(caffe::GetSolver<float,float>(solver_param));
//
//  solver->Restore(argv[1]);
//
//  shared_ptr<Net<float,float> > net = solver->net();
//  const vector<string>& blob_names = net->blob_names();
//
//  for (vector<string>::const_iterator it = blob_names.begin();
//      it != blob_names.end(); ++it) {
//    std::cout << *it << std::endl;
//    shared_ptr<Blob<float,float> > blob = net->blob_by_name(*it);
//    BlobProto blob_proto, hblob_proto;
//    blob->ToProto(&blob_proto, true);
//
//    hblob_proto.mutable_shape()->CopyFrom(blob_proto.shape());
//    const int data_size = blob_proto.data_size();
//
//    hblob_proto.mutable_half_data()->Reserve(data_size);
//    for (int i = 0; i < data_size; ++i) {
//      hblob_proto.mutable_half_data()->Set(i, (float16(blob_proto.data(i))).halfx());
//    }
//  }

  return 0;
}
