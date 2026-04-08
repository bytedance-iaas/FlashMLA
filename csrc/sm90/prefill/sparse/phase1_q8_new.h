#pragma once

#include "../../../params.h"

namespace sm90::fwd {

template<int D_QK, bool HAVE_TOPK_LENGTH>
void run_fwd_phase1_q8_sm90_new_kernel(const SparseAttnFwdQ8SM90NewParams& params);

}
