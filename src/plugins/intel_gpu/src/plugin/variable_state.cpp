// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/type/float16.hpp"
#include <memory>

namespace ov::intel_gpu {

VariableState::VariableState(const VariableStateInfo& info, RemoteContextImpl::Ptr context, std::shared_ptr<cldnn::ShapePredictor> shape_predictor)
    : VariableStateBase{info.m_id, context}
    , m_layout(info.m_layout)
    , m_user_specified_type(info.m_user_specified_type)
    , m_shape_predictor(shape_predictor)
    , m_prim_inst(info.m_release_variable_inst)
    , m_transpose_required(info.transpose_required)
    , m_initial_layout(info.m_layout) {
    update_device_buffer();
}

void VariableState::reset() {
    m_is_set = false;
    set_layout(m_initial_layout);
    for (auto& user : m_prim_inst) {
        if (const auto prim = user.lock(); prim) {
            prim->release_variable();
        }
    }
}

cldnn::memory::ptr VariableState::get_memory() const {
    return m_memory;
}

const cldnn::layout& VariableState::get_layout() const {
    return m_layout;
}

void VariableState::set_memory(const cldnn::memory::ptr& new_mem, const cldnn::layout& actual_layout) {
    GPU_DEBUG_TRACE_DETAIL << m_name << " : Update memory (Ptr : " << new_mem->buffer_ptr()
                           << ", layout : " << actual_layout.to_short_string() << ")" << std::endl;
    m_memory = new_mem;
    m_layout = actual_layout;
    actual_size = m_memory->size();
    update_device_buffer();
}

void VariableState::set_layout(const cldnn::layout& new_layout) {
    if (m_layout == new_layout)
        return;
    m_layout = new_layout;
    GPU_DEBUG_TRACE_DETAIL << m_name << " : " << "Update state layout to " << new_layout.to_short_string() << std::endl;
    update_device_buffer();
}

void VariableState::set_state(const ov::SoPtr<ov::ITensor>& state) {
    auto src_shape = state->get_shape();
    size_t src_rank = src_shape.size();
    cldnn::padding::DynamicDimsMask dynamic_pad_dims;
    for (size_t i = 0; i < src_rank; i++) {
        dynamic_pad_dims[i] = m_layout.data_padding._dynamic_dims_mask[i];
    }
    m_layout.data_padding = cldnn::padding(std::vector<ov::Dimension::value_type>(src_rank, 0),
                                           std::vector<ov::Dimension::value_type>(src_rank, 0),
                                           dynamic_pad_dims);
    auto src_stride = state->get_strides();
    for (size_t i = 0; i < src_rank; ++i) {
        src_stride[i] /= state->get_element_type().bitwidth() / 8;
    }
    m_layout.set_partial_shape(src_shape);
    update_device_buffer();

    if (actual_size == 0) {
        set();
        return;
    }

    // check whether the src tensor is padded
    std::vector<size_t> src_stride_no_pad(src_rank, 1);
    std::vector<ov::Dimension::value_type> upper_pad(src_rank, 0);
    std::vector<ov::Dimension::value_type> lower_pad(src_rank, 0);
    for (int32_t i = static_cast<int32_t>(src_stride.size()) - 1; i >= 0; --i) {
        if (i <= static_cast<int32_t>(src_stride.size()) - 2)
            src_stride_no_pad[i] = src_stride_no_pad[i + 1] * src_shape[i + 1];
        if (src_stride[i] != src_stride_no_pad[i]) {
            OPENVINO_ASSERT(src_stride[i] > src_stride_no_pad[i]);
            size_t padded_size = src_stride[i] / src_stride[i + 1];
            size_t non_padded_size = src_stride_no_pad[i] / src_stride_no_pad[i + 1];
            ov::Dimension::value_type pad_dim = i + 1;
            upper_pad[pad_dim] = static_cast<ov::Dimension::value_type>(padded_size) - static_cast<ov::Dimension::value_type>(non_padded_size);
        }
    }
    cldnn::padding src_padd = cldnn::padding(lower_pad, upper_pad, 0.f);
    auto src_fmt = cldnn::format::get_default_format(src_rank);
    auto src_layout = cldnn::layout(ov::PartialShape(src_shape), state->get_element_type(), src_fmt, src_padd);

    convert_and_copy(state._ptr.get(), m_memory, m_context->get_engine().get_service_stream(), src_layout, m_transpose_required);
    set();
}

void VariableState::update_device_buffer() {
    OPENVINO_ASSERT(m_context != nullptr, "m_context should not be null.");
    if (m_layout.is_dynamic() || m_layout.bytes_count() == 0) {
        m_shape_predictor->reset();
        m_memory.reset();
        actual_size = 0;
        return;
    }

    if (actual_size < m_layout.bytes_count()) {
        const auto alloc_type = m_context->get_engine().use_unified_shared_memory() ? cldnn::allocation_type::usm_device : cldnn::allocation_type::cl_mem;
        const auto current_buf_size = m_layout.get_padded_dims();
        ov::Shape current_shape(current_buf_size.begin(), current_buf_size.end());
        const auto alloc_shape = predict_shape(m_name, cldnn::layout(current_shape, m_layout.data_type, m_layout.format), *m_shape_predictor);
        const auto alloc_layout = cldnn::layout(alloc_shape, m_layout.data_type, m_layout.format);
        m_memory = m_context->get_engine().allocate_memory(alloc_layout, alloc_type, false);
        actual_size = std::max(actual_size, alloc_layout.bytes_count());
    }

    OPENVINO_ASSERT(m_memory != nullptr, "m_memory is nullptr!!!");
    m_memory = m_context->get_engine().reinterpret_buffer(*m_memory, m_layout);
}

ov::element::Type VariableState::get_user_specified_type() const {
    return m_user_specified_type != ov::element::dynamic ? m_user_specified_type : ov::element::Type(m_layout.data_type);
}

void VariableState::set_state_from_memory_slice(const cldnn::memory::ptr& src_5d_mem,
                                                size_t token_position,
                                                const ov::Shape& all_states_shape) {
    // Supports both 5D (linear states: [B, T, H, K, V]) and 4D (conv states: [B, T, D, KS])
    OPENVINO_ASSERT(all_states_shape.size() == 5 || all_states_shape.size() == 4,
                    "Expected 4D or 5D shape for all_states, got ", all_states_shape.size());

    const size_t B = all_states_shape[0];
    const size_t T = all_states_shape[1];

    // Compute per-token state elements and target shape from remaining dims
    size_t state_elems = 1;
    ov::Shape state_shape = {B};
    for (size_t i = 2; i < all_states_shape.size(); ++i) {
        state_elems *= all_states_shape[i];
        state_shape.push_back(all_states_shape[i]);
    }

    OPENVINO_ASSERT(token_position < T, "token_position ", token_position, " >= T ", T);

    auto src_et = src_5d_mem->get_layout().data_type;
    size_t src_elem_bytes = ov::element::Type(src_et).size();
    size_t dst_elem_bytes = ov::element::Type(m_layout.data_type).size();

    // Ensure variable's device buffer is allocated for state_shape
    m_layout.set_partial_shape(state_shape);
    update_device_buffer();

    if (actual_size == 0 || m_layout.bytes_count() == 0) {
        set();
        return;
    }

    OPENVINO_ASSERT(m_memory != nullptr, "[GPU] Variable memory is null after update_device_buffer");

    auto& stream = m_context->get_engine().get_service_stream();

    if (src_elem_bytes == dst_elem_bytes) {
        // Same data type — direct GPU-to-GPU copy with offset
        if (B == 1) {
            size_t src_offset = token_position * state_elems * src_elem_bytes;
            size_t copy_bytes = state_elems * src_elem_bytes;
            m_memory->copy_from(stream, *src_5d_mem, src_offset, 0, copy_bytes, true);
        } else {
            for (size_t b = 0; b < B; ++b) {
                size_t src_offset = (b * T * state_elems + token_position * state_elems) * src_elem_bytes;
                size_t dst_offset = b * state_elems * dst_elem_bytes;
                m_memory->copy_from(stream, *src_5d_mem, src_offset, dst_offset, state_elems * src_elem_bytes, true);
            }
        }
    } else {
        // Type mismatch (e.g. f32 output → f16 variable): copy slice GPU→CPU, convert, upload
        size_t slice_bytes = B * state_elems * src_elem_bytes;
        std::vector<uint8_t> host_buf(slice_bytes);

        for (size_t b = 0; b < B; ++b) {
            size_t src_offset = (b * T * state_elems + token_position * state_elems) * src_elem_bytes;
            size_t dst_offset = b * state_elems * src_elem_bytes;
            src_5d_mem->copy_to(stream, host_buf.data(), src_offset, dst_offset, state_elems * src_elem_bytes, true);
        }

        // Create a temporary CPU tensor wrapping the host buffer, then use convert_and_copy
        auto src_ov_et = ov::element::Type(src_et);
        auto dst_ov_et = ov::element::Type(m_layout.data_type);
        ov::Tensor dst_host(dst_ov_et, state_shape);
        size_t total_elems = B * state_elems;

        // Simple element-wise conversion
        if (src_ov_et == ov::element::f32 && dst_ov_et == ov::element::f16) {
            auto* sp = reinterpret_cast<const float*>(host_buf.data());
            auto* dp = reinterpret_cast<ov::float16*>(dst_host.data());
            for (size_t i = 0; i < total_elems; ++i)
                dp[i] = ov::float16(sp[i]);
        } else if (src_ov_et == ov::element::f16 && dst_ov_et == ov::element::f32) {
            auto* sp = reinterpret_cast<const ov::float16*>(host_buf.data());
            auto* dp = reinterpret_cast<float*>(dst_host.data());
            for (size_t i = 0; i < total_elems; ++i)
                dp[i] = float(sp[i]);
        } else {
            OPENVINO_THROW("[GPU] Unsupported type conversion in set_state_from_memory_slice: ",
                           src_ov_et, " -> ", dst_ov_et);
        }

        m_memory->copy_from(stream, dst_host.data(), true);
    }

    set();
}

void VariableState::trim_state(size_t trim_amount, size_t axis) {
    auto shape = m_layout.get_shape();
    OPENVINO_ASSERT(axis < shape.size(),
                    "[GPU] trim_state: axis ", axis, " >= rank ", shape.size());
    OPENVINO_ASSERT(shape[axis] >= trim_amount,
                    "[GPU] trim_state: trim_amount ", trim_amount,
                    " > dim[", axis, "] = ", shape[axis]);

    shape[axis] -= trim_amount;
    m_layout.set_partial_shape(shape);
    update_device_buffer();   // reinterpret_buffer only (no alloc/copy)
    set();
}

ov::SoPtr<ov::ITensor> VariableState::get_state() const {
    if (m_memory == nullptr) {
        const auto& pshape = m_layout.get_partial_shape();
        const auto& shape = get_tensor_shape(pshape);
        return m_context->create_host_tensor(get_user_specified_type(), shape);
    }

    auto tensor = m_context->create_host_tensor(get_user_specified_type(), m_memory->get_layout().get_shape());

    convert_and_copy(m_memory, tensor._ptr.get(), m_context->get_engine().get_service_stream());

    return tensor;
}

}  // namespace ov::intel_gpu
