#include "cntk/include/CNTKLibrary.h"
#include "cntk/include/CNTKLibraryC.h"
#include "cntk/include/CNTKLibraryInternals.h"
#include "cntk/include/Eval.h"
#include "cntk/include/HalfConverter.hpp"

#include "../common/common.h"

namespace horovod {
namespace cntk {

using namespace horovod::common;
using namespace CNTK;

class TensorUtil {
public:
  static const DataType GetDType(NDArrayView* tensor);
  static const horovod::common::TensorShape GetShape(NDArrayView* tensor);
  static const void* GetData(NDArrayView* tensor);
  static int64_t GetSize(NDArrayView* tensor);
  static int GetDevice(NDArrayView* tensor);

  static NDArrayView* New(int device, int dtype);
  static void Free(NDArrayView* tensor);
  // static void ResizeNd(NDArrayView* tensor, int nDimension, int64_t* size);
  // static void Copy(NDArrayView* output, NDArrayView* tensor);
  // static void DivideTensorInPlace(NDArrayView* tensor, int value);
private:
  static const size_t kFloat32Size = 4;
  static const size_t kFloat64Size = 8;
  static const size_t kFloat16Size = 2;
  static const size_t kUInt8Size = 1;
  static const size_t kInt32Size = 4;
  static const size_t kInt8Size = 1;
  static const size_t kInt64Size = 8;
};

} //namsepace cntk
} // namespace horovod