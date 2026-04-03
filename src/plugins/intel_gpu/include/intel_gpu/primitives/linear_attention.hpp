// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"
#include "openvino/op/linear_attn.hpp"
#include "openvino/op/util/variable.hpp"
#include <vector>

namespace cldnn {

using LinearAttention = ov::op::LinearAttention;

/// @brief linear_attention primitive
/// @details Performs linear_attention
struct linear_attention : public primitive_base<linear_attention> {
    CLDNN_DECLARE_PRIMITIVE(linear_attention)

    linear_attention() : primitive_base("", {}) {}

    /// @brief Constructs linear_attention primitive / layer (no variable).
    ///
    /// @param id                 An identifier of new primitive.
    /// @param inputs             A list of Input primitive ids (inputs).
    linear_attention(const primitive_id& id,
            const std::vector<input_info>& inputs)
        : primitive_base(id, inputs) {
    }

    /// @brief Constructs linear_attention primitive / layer (variable-aware).
    ///
    /// @param id                 An identifier of new primitive.
    /// @param inputs             A list of Input primitive ids (inputs).
    /// @param variable_info      Variable info for recurrent state.
    /// @param snapshot_all_states Whether to output per-token intermediate states.
    linear_attention(const primitive_id& id,
            const std::vector<input_info>& inputs,
            const ov::op::util::VariableInfo& variable_info,
            bool snapshot_all_states = false)
        : primitive_base(id, inputs),
          variable_info(variable_info),
          snapshot_all_states(snapshot_all_states) {
    }

    ov::op::util::VariableInfo variable_info;
    bool snapshot_all_states = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, snapshot_all_states);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<linear_attention>::save(ob);
        ov::element::Type_t data_type = variable_info.data_type;
        ob << variable_info.variable_id;
        ob << variable_info.data_shape;
        ob << make_data(&data_type, sizeof(ov::element::Type_t));
        ob << snapshot_all_states;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<linear_attention>::load(ib);
        ov::PartialShape data_shape;
        ov::element::Type_t data_type = ov::element::Type_t::dynamic;
        std::string variable_id;
        ib >> variable_id;
        ib >> data_shape;
        ib >> make_data(&data_type, sizeof(ov::element::Type_t));
        variable_info = {data_shape, data_type, variable_id};
        ib >> snapshot_all_states;
    }
};

}  // namespace cldnn
