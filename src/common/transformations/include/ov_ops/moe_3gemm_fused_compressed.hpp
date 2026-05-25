// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ov_ops/moe_compressed.hpp"

namespace ov::op::internal {

/// \brief Fused 3-GEMM compressed MOE op with GPU runtime layout.
class TRANSFORMATIONS_API MOE3GemmFusedCompressed : public MOECompressed {
public:
    OPENVINO_OP("MOE3GemmFusedCompressed", "gpu_opset", MOECompressed);

    MOE3GemmFusedCompressed() = default;
    MOE3GemmFusedCompressed(const OutputVector& args, const MOECompressed::Config& config);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace ov::op::internal
