// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/linear_attn.hpp"

#include "dimension_util.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"

namespace {

// Validates input rank and type for a node input.
// We consider that dynamic rank/type are always valid case.
// Empty {} means any rank/type
inline void input_check(const ov::Node* node,
                        size_t idx,
                        const std::string_view input_name,
                        std::initializer_list<ov::Rank>&& allowed_ranks,
                        const std::vector<ov::element::Type>& allowed_types) {
    using namespace ov;
    using namespace ov::util;
    using namespace ov::element;

    const auto& rank = node->get_input_partial_shape(idx).rank();
    const auto& tp = node->get_input_element_type(idx);

    auto rank_check = [&](const Rank& rank) {
        return rank.is_dynamic() || empty(allowed_ranks) || is_rank_compatible_any_of(rank.get_length(), allowed_ranks);
    };

    auto type_check = [&](const Type& type) {
        auto it = std::find(allowed_types.begin(), allowed_types.end(), tp);
        return type.is_dynamic() || allowed_types.empty() || it != allowed_types.end();
    };

    NODE_VALIDATION_CHECK(node,
                          rank_check(rank),
                          "Rank of `",
                          input_name,
                          "` input should be in [dynamic, ",
                          join(allowed_ranks),
                          "] list, but it is ",
                          rank,
                          ".");

    NODE_VALIDATION_CHECK(node,
                          type_check(tp),
                          "Element type of `",
                          input_name,
                          "` input should be in [dynamic, ",
                          join(allowed_types),
                          "] list, but it is ",
                          tp,
                          ".");
}
}  // namespace

namespace ov {
namespace op {

LinearAttention::LinearAttention(const ov::OutputVector& args) : ov::op::Op(args) {
    constructor_validate_and_infer_types();
}

LinearAttention::LinearAttention(const ov::OutputVector& args, const std::shared_ptr<ov::op::util::Variable>& variable)
    : ov::op::Op(args) {
    m_variable = variable;
    constructor_validate_and_infer_types();
}

LinearAttention::LinearAttention(const ov::OutputVector& args,
                                 const std::shared_ptr<ov::op::util::Variable>& variable,
                                 bool snapshot_all_states)
    : ov::op::Op(args),
      m_snapshot_all_states(snapshot_all_states) {
    m_variable = variable;
    constructor_validate_and_infer_types();
}

bool LinearAttention::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(LinearAttention_visit_attributes);

    visitor.on_attribute("variable_id", m_variable);
    if (m_variable) {
        auto variable_info = m_variable->get_info();
        visitor.on_attribute("variable_type", variable_info.data_type);
        visitor.on_attribute("variable_shape", variable_info.data_shape);
        m_variable->update(variable_info);
    }
    visitor.on_attribute("snapshot_all_states", m_snapshot_all_states);

    return true;
}

void LinearAttention::validate_and_infer_types() {
    OV_OP_SCOPE(LinearAttention_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 6,
                          "LinearAttention expects 6 inputs, but it has ",
                          get_input_size());

    // format: Node*, input_idx, name, {rank_list}, {type_list}
    input_check(this, 0, "query", {4}, {});
    input_check(this, 1, "key", {4}, {});
    input_check(this, 2, "value", {4}, {});
    input_check(this, 3, "beta", {3}, {});
    input_check(this, 4, "g", {3}, {});
    input_check(this, 5, "initial_states", {4}, {});

    // value head_size may be not same with key, output uses value head count
    const auto& q_ps = get_input_partial_shape(0);
    const auto& v_ps = get_input_partial_shape(2);
    const auto& h_ps = get_input_partial_shape(5);

    ov::PartialShape out_ps = v_ps;
    if (out_ps.rank().is_static() && q_ps.rank().is_static() && out_ps.rank().get_length() == 4 && q_ps.rank().get_length() == 4) {
        out_ps[0] = q_ps[0];
        out_ps[1] = q_ps[1];
    }
    set_output_type(0, get_input_element_type(0), out_ps);
    set_output_type(1, get_input_element_type(5), h_ps);

    // Output 2: per-token intermediate recurrent states [B, T, H_v, K, V]
    // Used by MTP speculative decoding to select state at num_accepted position.
    if (m_snapshot_all_states) {
        // h_ps = [B, H_v, K, V], q_ps = [B, T, H_q, K]
        // all_states = [B, T, H_v, K, V]
        ov::PartialShape snap_ps = h_ps;  // start from [B, H_v, K, V]
        if (snap_ps.rank().is_static() && q_ps.rank().is_static()) {
            // Insert T dimension at position 1: [B, T, H_v, K, V]
            ov::PartialShape snap_5d;
            snap_5d.push_back(q_ps[0]);   // B
            snap_5d.push_back(q_ps[1]);   // T (seq_len)
            for (size_t i = 1; i < snap_ps.size(); i++) {
                snap_5d.push_back(snap_ps[i]);  // H_v, K, V
            }
            snap_ps = snap_5d;
        }
        set_output_type(2, get_input_element_type(5), snap_ps);
    }
}

std::shared_ptr<ov::Node> LinearAttention::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    if (m_variable) {
        return std::make_shared<LinearAttention>(new_args, m_variable, m_snapshot_all_states);
    }
    return std::make_shared<LinearAttention>(new_args);
}

void LinearAttention::set_out_type(int index, const ov::element::Type& output_type) {
    OPENVINO_ASSERT(index < 2, "Output index should be 0 or 1, but got " + std::to_string(index));
    m_output_type[index] = output_type;
}

}  // namespace op
}  // namespace ov
