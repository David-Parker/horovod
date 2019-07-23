import os
import cntk as C

from horovod.common.util import get_ext_suffix
from horovod.common.basics import HorovodBasics as _HorovodBasics
_basics = _HorovodBasics(__file__, 'mpi_lib')

# import basic methods
init = _basics.init
shutdown = _basics.shutdown
size = _basics.size
local_size = _basics.local_size
rank = _basics.rank
local_rank = _basics.local_rank
mpi_threads_supported = _basics.mpi_threads_supported

dll_path = os.path.join(os.path.dirname(__file__),
                        'mpi_lib' + get_ext_suffix())
MPI_CNTK_LIB_CTYPES = ctypes.CDLL(dll_path, ctypes.RTLD_GLOBAL)


def allreduce(tensor):
    # implement allreduce by calling respective cpp library

def broadcast(tensor):
    # implement broadcast method by calling respective cpp library