#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

// FusedConv kernel: fuses Gather(beam_idx) + Concat + DepthwiseConv1D + SiLU + Slice
//
// Inputs:
//   INPUT0 (input):       [B, CONV_DIM, S]
//   INPUT1 (conv_weight): [CONV_DIM, KERNEL_SIZE]
//   INPUT2 (beam_idx):    [B]
//   INPUT3 (state_in):    [B, CONV_DIM, KERNEL_SIZE]
//
// Outputs:
//   OUTPUT  (output):     [B, CONV_DIM, S]
//   OUTPUT1 (state_out):  [B, CONV_DIM, KERNEL_SIZE]
//   OUTPUT2 (all_states): [B, S, CONV_DIM, KERNEL_SIZE] (SNAPSHOT_MODE only)
//
// Dispatch: global = {batch, conv_dim, 1}, local = {1, WG_SIZE, 1}

KERNEL(fused_conv_ref)(
    __global INPUT0_TYPE* input,
    __global INPUT1_TYPE* conv_weight,
    __global INPUT2_TYPE* beam_idx,
    __global INPUT3_TYPE* state_in,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* state_out,
#ifdef SNAPSHOT_MODE
    __global OUTPUT2_TYPE* all_states,
#endif
    int seq_len)
{
    const int b  = get_global_id(0);
    const int ch = get_global_id(1);

    if (ch >= CONV_DIM)
        return;

    // 1. Beam search reorder: read state from beam_idx[b] source batch
    const int src_b = convert_int(beam_idx[b]);

    // 2. Load state (KERNEL_SIZE values)
    float state[KERNEL_SIZE];
    const int state_in_base = src_b * CONV_DIM * KERNEL_SIZE + ch * KERNEL_SIZE;
    for (int k = 0; k < KERNEL_SIZE; k++)
        state[k] = convert_float(state_in[state_in_base + k]);

    // 3. Load conv weight
    float w[KERNEL_SIZE];
    const int w_base = ch * KERNEL_SIZE;
    for (int k = 0; k < KERNEL_SIZE; k++)
        w[k] = convert_float(conv_weight[w_base + k]);

    // 4. For each sequence position: depthwise conv + SiLU
    const int io_base = b * CONV_DIM * seq_len + ch * seq_len;
    for (int s = 0; s < seq_len; s++) {
        float x_new = convert_float(input[io_base + s]);

        // Conv window = [state[1], state[2], ..., state[K-1], x_new]
        // This corresponds to concatenating state with input and applying valid conv
        float acc = 0.0f;
        for (int k = 0; k < KERNEL_SIZE - 1; k++)
            acc += state[k + 1] * w[k];
        acc += x_new * w[KERNEL_SIZE - 1];

        // SiLU activation: x * sigmoid(x)
        float sig = native_recip(1.0f + native_exp(-acc));
        output[io_base + s] = TO_OUTPUT_TYPE(acc * sig);

        // Shift state left, append new input
        for (int k = 0; k < KERNEL_SIZE - 1; k++)
            state[k] = state[k + 1];
        state[KERNEL_SIZE - 1] = x_new;

        // F16 rounding at token boundary: match sequential single-token precision.
        // In sequential mode (seq_len=1), state is stored as f16 after each token and
        // reloaded as f16→f32 for the next call. In batch mode (seq_len>1), state stays
        // in f32 registers across tokens. This f32→f16→f32 round-trip ensures batch
        // results are numerically identical to sequential processing.
        // Skip the last token since it's followed by the final state store anyway.

#ifdef SNAPSHOT_MODE
        // Save per-token conv state snapshot: all_states[b, s, ch, 0..KERNEL_SIZE-1]
        // Layout: [B, S, CONV_DIM, KERNEL_SIZE] — contiguous in KERNEL_SIZE
        const int snap_base = ((b * seq_len + s) * CONV_DIM + ch) * KERNEL_SIZE;
        for (int k = 0; k < KERNEL_SIZE; k++)
            all_states[snap_base + k] = TO_OUTPUT2_TYPE(state[k]);
        // Load back from snapshot for memory-based f16 round-trip (prevents compiler
        // from optimizing away the precision loss and ensures snapshot data matches
        // what subsequent tokens use — critical for kernel-snapshot restore).
        if (s < seq_len - 1) {
            for (int k = 0; k < KERNEL_SIZE; k++)
                state[k] = convert_float(all_states[snap_base + k]);
        }
#else
        if (s < seq_len - 1) {
            for (int k = 0; k < KERNEL_SIZE; k++)
                state[k] = convert_float(TO_OUTPUT1_TYPE(state[k]));
        }
#endif
    }

    // 5. Write back updated state
    const int state_out_base = b * CONV_DIM * KERNEL_SIZE + ch * KERNEL_SIZE;
    for (int k = 0; k < KERNEL_SIZE; k++)
        state_out[state_out_base + k] = TO_OUTPUT1_TYPE(state[k]);
}
