#include "../phase1_q8_new.h"
#include "../phase1_new.cuh"

namespace sm90::fwd {

template void run_fwd_phase1_q8_sm90_new_kernel<512, false>(const SparseAttnFwdQ8SM90NewParams& params);
template void run_fwd_phase1_q8_sm90_new_kernel<512, true>(const SparseAttnFwdQ8SM90NewParams& params);

}
