#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sparse_fwd.h"
#include "sparse_decode.h"
#include "dense_decode.h"
#include "dense_fwd.h"

namespace py = pybind11;

static std::vector<at::Tensor> sparse_prefill_q8kv8_fwd_py(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    float sm_scale,
    int d_v,
    const at::Tensor &q_scale,
    const at::Tensor &kv_scale,
    py::object attn_sink_obj,
    py::object topk_length_obj) {
    std::optional<at::Tensor> attn_sink = std::nullopt;
    std::optional<at::Tensor> topk_length = std::nullopt;

    if (!attn_sink_obj.is_none()) {
        attn_sink = attn_sink_obj.cast<at::Tensor>();
    }
    if (!topk_length_obj.is_none()) {
        topk_length = topk_length_obj.cast<at::Tensor>();
    }

    return sparse_attn_prefill_q8kv8_interface(
        q,
        kv,
        indices,
        sm_scale,
        d_v,
        q_scale,
        kv_scale,
        attn_sink,
        topk_length);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("sparse_decode_fwd", &sparse_attn_decode_interface);
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
    m.def(
        "sparse_prefill_q8kv8_fwd",
        &sparse_prefill_q8kv8_fwd_py,
        py::arg("q"),
        py::arg("kv"),
        py::arg("indices"),
        py::arg("sm_scale"),
        py::arg("d_v"),
        py::arg("q_scale"),
        py::arg("kv_scale"),
        py::arg("attn_sink") = py::none(),
        py::arg("topk_length") = py::none());
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
}
