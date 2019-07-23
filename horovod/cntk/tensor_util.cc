#include "tensor_util.h"

#include "cntk/include/CNTKLibrary.h"

namespace horovod {
namespace cntk {

using namespace CNTK;

// Tyler: Only worry about floats and doubles

// Define all types for TensorUtil.
const DataType TensorUtil::GetDType(NDArrayView* tensor) {
  switch (tensor->GetDataType()) {
  case CNTK::DataType::Float:
    return DataType::HOROVOD_FLOAT32;
  case CNTK::DataType::Double:
    return DataType::HOROVOD_FLOAT64;
  // case CNTK::DataType::Float16:
  //   return DataType::HOROVOD_FLOAT16;
  // case CNTK::DataType::Int8:
  //   return DataType::HOROVOD_INT8;
  // case CNTK::DataType::Int16:
  //   return DataType::HOROVOD_INT16;
  default:
    throw std::logic_error("GetDType: Type " + CNTK::DataType::Unknown +
                           " is not supported in MPI mode.");
  }
}

// Return shape of tensor (similar to TShape)
const TensorShape TensorUtil::GetShape(NDArrayView* tensor) {
  TensorShape shape;
  std::vector<size_t> nd_shape = tensor->Shape().Dimensions();
  for (int idx = 0; idx < (int)nd_shape.size(); idx++) {
    shape.AddDim(nd_shape[idx]);
  }
  return shape;
}

// Return data of tensor
const void* TensorUtil::GetData(NDArrayView* tensor) {
  // The following returns an error:
  // return tensor->data().dptr<void>();
  switch (tensor->GetDataType()) {
  case CNTK::DataType::Float:
    return static_cast<void*>(tensor->WritableDataBuffer<float>());
  case CNTK::DataType::Double:
    return static_cast<void*>(tensor->WritableDataBuffer<double>());
  default:
    const std::string d_type = CNTK::DataTypeName(tensor->GetDataType());
    throw std::logic_error("Type " + d_type + " is not supported in MPI mode.");
  }
}

// Return size of tensor in bytes
int64_t TensorUtil::GetSize(NDArrayView* tensor) {
  int64_t element_size = 0;
  switch (tensor->GetDataType()) {
  case CNTK::DataType::Float:
    element_size = kFloat32Size;
  case CNTK::DataType::Double:
    element_size = kFloat64Size;
  default:
    const std::string d_type = CNTK::DataTypeName(tensor->GetDataType());
    throw std::logic_error("Type " + d_type + " is not supported in MPI mode.");
  }
  // Tyler: Get the total number of elements in the tensor and multiple element_size
  return (int64_t)(tensor->Shape().TotalSize()) * element_size;
}

// If Tensor on GPU, return device id
// Otherwise return CPU_DEVICE_ID (-1)
int TensorUtil::GetDevice(NDArrayView* tensor) {
  CNTK::DeviceKind dev_mask = tensor->Device().Type();
  if (dev_mask == CNTK::DeviceKind::GPU)
    return tensor->Device().Id();
  return CPU_DEVICE_ID;
}

// Tyler: Assumes the datatype is float for all cases
// Returns pointer to newly created NDArrayView
// If dev_id equal to CPU_DEVICE_ID, construct Tensor on CPU
// Otherwise construct on GPU
NDArrayView* TensorUtil::New(int device, int dtype) {
  if (device == CPU_DEVICE_ID) {
    NDArrayView* my_array = new NDArrayView(AsDataType<float>(), CNTK::NDShape::Unknown(), CNTK::DeviceDescriptor::CPUDevice());
    return my_array;
  } else {
    CNTK::DeviceDescriptor dev_cpu = CNTK::DeviceDescriptor::GPUDevice(device);
    NDArrayView* my_array =
        new NDArrayView(AsDataType<float>(), CNTK::NDShape::Unknown(), CNTK::DeviceDescriptor::CPUDevice());
    return my_array;
  }
}

void TensorUtil::Free(NDArrayView* tensor) { delete tensor; }

// TODO: Reshape utils
// // Resize tensor to nDimension with length size[i] in dimension i
// void TensorUtil::ResizeNd(NDArrayView* tensor, int nDimension, int64_t* size) {
//   void* temp_out;
//   MXNDArrayViewReshape64(tensor, nDimension, size, false, &temp_out);
//   tensor = static_cast<NDArrayView*>(temp_out);
// }

// // Copy from tensor to output
// void TensorUtil::Copy(NDArrayView* output, NDArrayView* tensor) {
//   if (tensor->shape() != output->shape())
//     output->ReshapeAndAlloc(tensor->shape());
//   CopyFromTo(*tensor, output, 0);
// }

// // Elementwise division of tensor by value in-place
// void TensorUtil::DivideTensorInPlace(NDArrayView* tensor, int value) {
//   *tensor /= value;
// }

} // namespace mxnet
} // namespace horovod