#ifdef USE_CUDNN
#include "caffe/util/cudnn.hpp"

namespace caffe {
namespace cudnn {

float dataType<float>::oneval = 1.0;
float dataType<float>::zeroval = 0.0;
const void* dataType<float>::one =
    static_cast<void *>(&dataType<float>::oneval);
const void* dataType<float>::zero =
    static_cast<void *>(&dataType<float>::zeroval);

double dataType<double>::oneval = 1.0;
double dataType<double>::zeroval = 0.0;
const void* dataType<double>::one =
    static_cast<void *>(&dataType<double>::oneval);
const void* dataType<double>::zero =
    static_cast<void *>(&dataType<double>::zeroval);

float dataType<float16>::oneval = 1.0f;
float dataType<float16>::zeroval = 0.0f;

const void* dataType<float16>::one =
    static_cast<void *>(&dataType<float16>::oneval);
const void* dataType<float16>::zero =
    static_cast<void *>(&dataType<float16>::zeroval);

}  // namespace cudnn
}  // namespace caffe
#endif
