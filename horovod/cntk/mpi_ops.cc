#include <stdio.h>
#include "../common/operations.h"
#include "CNTKLibrary.h"

namespace horovod {
namespace cntk {

namespace {

void func()
{
    CNTK::NDShape* shape = new CNTK::NDShape(2, 2);

    auto tSize = shape->TotalSize();

    printf("Shape is scalar: %s with total size %lu\n", shape->IsScalar() ? "true" : "false", tSize);
}
} // namespace
} // namespace cntk
} // namespace horovod