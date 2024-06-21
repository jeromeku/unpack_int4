#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>


TORCH_LIBRARY(unpack_int4, m) {
  m.def("unpack_int4_packed(Tensor packed_w, int innerKTiles) -> Tensor");
}
