#ifndef __LINDIT_HPP__
#define __LINDIT_HPP__

#include "ggml_extend.hpp"
#include "model.h"

// TODO: estimate actual size
#define LINDIT_GRAPH_SIZE 10240

__STATIC_INLINE__ struct ggml_tensor* modulate_gated(struct ggml_context* ctx,
                                                     struct ggml_tensor* x,
                                                     struct ggml_tensor* shift,
                                                     struct ggml_tensor* scale,
                                                     struct ggml_tensor* gate) {
    // x: [N, L, C]
    // scale: [N, C]
    // shift: [N, C]
    // gate:  [N, C]
    scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);  // [N, 1, C]
    shift = ggml_reshape_3d(ctx, shift, shift->ne[0], 1, shift->ne[1]);  // [N, 1, C]
    gate  = ggml_reshape_3d(ctx, shift, gate->ne[0], 1, gate->ne[1]);    // [N, 1, C]
    x     = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
    x     = ggml_add(ctx, x, shift);
    x     = ggml_add(ctx, x, ggml_mul(ctx, x, gate));
    return x;
}

__STATIC_INLINE__ std::array<std::vector<struct ggml_tensor*>, 2> chunk_table(struct ggml_context* ctx, struct ggml_tensor* m) {
    // shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn
    int64_t offset                = m->nb[1] * m->ne[1];
    struct ggml_tensor* shift_msa = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], 0);           // [hidden_size, 1]
    struct ggml_tensor* scale_msa = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 1);  // [hidden_size, 1]
    struct ggml_tensor* gate_msa  = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 2);  // [hidden_size, 1]
    struct ggml_tensor* shift_ffn = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 3);  // [hidden_size, 1]
    struct ggml_tensor* scale_ffn = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 4);  // [hidden_size, 1]
    struct ggml_tensor* gate_ffn  = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 5);  // [hidden_size, 1]

    return {{shift_msa, scale_msa, gate_msa}, {shift_ffn, scale_ffn, gate_ffn}};
}

struct LinearAttentionHead : public GGMLBlock {
public:
    LinearAttentionHead(int64 dim, bool qkv_bias = false) {
        blocks["to_q"]     = std::shared_ptr<Linear>(new Linear(dim, qkv_bias));
        blocks["to_k"]     = std::shared_ptr<Linear>(new Linear(dim, qkv_bias));
        blocks["to_v"]     = std::shared_ptr<Linear>(new Linear(dim, qkv_bias));
        blocks["to_out.0"] = std::shared_ptr<Linear>(new Linear(dim, true));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x_in, struct ggml_tensor* x, std::vector<struct ggml_tensor*> shifts_scales_gate) {
        // linear attention: out = relu(q)*(relu(transpose(k))*v))
        Linear* to_q   = params["to_q"];
        Linear* to_k   = params["to_k"];
        Linear* to_v   = params["to_v"];
        Linear* to_out = params["to_out.0"];
        // x:    [N, seq_len, hidden_size]
        // x_in: [N, seq_len, hidden_size] (=x for self attn)

        struct ggml_tensor* q = to_q->forward(ctx, x);     // [N, seq_len, dim]
        struct ggml_tensor* k = to_k->forward(ctx, x_in);  // [N, seq_len, dim]
        struct ggml_tensor* v = to_v->forward(ctx, x_in);  // [N, seq_len, dim]

        // TODO: is that the right OP?
        k = ggml_transpose(ctx, k);  // [N, dim, seq_len]

        q = ggml_relu_inplace(ctx, q);
        k = ggml_relu_inplace(ctx, k);

        struct ggml_tensor* kv  = ggml_mul_mat_2d(ctx, k, v);   // [N, dim, dim]
        struct ggml_tensor* out = ggml_mul_mat_2d(ctx, q, kv);  // [N, seq_len, dim]

        out = to_out->forward(ctx, out);  // [N, seq_len, hidden_size]

        out = modulate_gated(ctx, out, shifts_scales_gate[0], shifts_scales_gate[1], shifts_scales_gate[2]);
        return out;
    }
};

struct MixFFN : public GGMLBlock {
public:
    MixFFN(int64 hidden_size = 1152, int64 ffn_dim = 2880) {
        blocks["conv_inverted"] = std::shared_ptr<Conv2d>(new Conv2d(hidden_size, ffn_dim * 2, 1));
        blocks["conv_depth"]    = std::shared_ptr<Conv2d>(new Conv2d(ffn_dim * 2, ffn_dim * 2, 3));
        blocks["conv_point"]    = std::shared_ptr<Conv2d>(new Conv2d(ffn_dim, hidden_size, 1, {1, 1}, {0, 0}, {1, 1}, false));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        struct Conv2d* c_i = params["conv_inverted"];
        struct Conv2d* c_d = params["conv_depth"];
        struct Conv2d* c_p = params["conv_point"];

        // x: [N, W*H, hidden_size] ?

        // TODO reshape x for conv? (maybe [N,hidden_size,W,H])
        // dim up
        x = c_i->forward(ctx, x);  // [N, W*H, ffn_dim*2] ??
        // 3x3 conv
        x = c_d->forward(ctx, x);  // [N, W*H, ffn_dim*2] ???

        // chunk in two halves
        // TODO: views are probably wrong
        // TODO: cont()?
        ggml_tensor* gate = ggml_view_4d(ctx, x, x->ne[0], x->ne[1], x->ne[2] / 2, x->ne[3],
                                         x->nb[1], x->nb[2], x->nb[3], x->nb[2] / 2);  // [N, W*H, ffn_dim] ??
        x                 = ggml_view_4d(ctx, x, x->ne[0], x->ne[1], x->ne[2] / 2, x->ne[3],
                                         x->nb[1], x->nb[2], x->nb[3], 0);  // [N, W*H, ffn_dim] ??
        // non-linear
        gate = ggml_silu_inplace(ctx, gate);
        x    = ggml_mul_inplace(ctx, x, gate);

        // project back to hidden dim
        x = c_p->forward(ctx, x);  // [N, W*H, hidden_size] ?

        // TODO: re-reshape? (x needs to be same shape as input)
        return x;
    }
};

struct LinearTransformerBlock : public GGMLBlock {
protected:
    int64_t hidden_size;

    void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "") {
        ggml_type wtype             = (tensor_types.find(prefix + "scale_shift_table") != tensor_types.end()) ? tensor_types[prefix + "scale_shift_table"] : GGML_TYPE_F32;
        params["scale_shift_table"] = ggml_new_tensor_2d(ctx, wtype, hidden_size, 6);
    }

public:
    LinearTransformerBlock(int64_t dim) {
        // TODO: init
        blocks["attn1"] = std::shared_ptr<LinearAttentionHead>(new LinearAttentionHead(dim));
        blocks["attn2"] = std::shared_ptr<LinearAttentionHead>(new LinearAttentionHead(dim, true));

        blocks["ff"] = std::shared_ptr<MixFFN>(new MixFFN(dim));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* y, struct ggml_tensor* t) {
        LinearAttentionHead* self_attn  = blocks["attn1"];
        LinearAttentionHead* cross_attn = blocks["attn2"];
        MixFFN* ff                      = blocks["ff"];

        std::vector<std::vector<struct ggml_tensor*>> modulations = chunk_table(ctx, params["scale_shift_table"]);  // {{shift_msa, scale_msa, gate_msa}, {shift_ffn, scale_ffn, gate_ffn}}

        struct ggml_tensor* context = ggml_concat_2d(ctx, x, t, 2);  // [N, seq_len, hidden_size]

        x = self_attn->forward(ctx, x, x, modulations[0]);
        x = cross_attn->forward(ctx, x, y, modulations[1]);
        x = ff->forward(ctx, x);

        return x;
    }
};

struct LinDiT : public GGMLBlock {
protected:
    int depth           = 28;
    int64_t hidden_size = 1152;

public:
    LinDiT(std::map<std::string, enum ggml_type>& tensor_types) {
        // TODO global blocks
        for (int i = 0; i < depth; i++) {
            blocks["transformer_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new LinearTransformerBlock(hidden_size));
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* t,
                                struct ggml_tensor* y        = NULL,
                                struct ggml_tensor* context  = NULL,
                                std::vector<int> skip_layers = std::vector<int>()) {
        // TODO
    }
};

struct LinDiTRunner : public GGMLRunner {
    LinDiT model;
    static std::map<std::string, enum ggml_type> empty_tensor_types;

    MMDiTRunner(ggml_backend_t backend,
                std::map<std::string, enum ggml_type>& tensor_types = empty_tensor_types,
                const std::string prefix                            = "")
        : GGMLRunner(backend), model(tensor_types) {
        model.init(params_ctx, tensor_types, prefix);
    }

    std::string get_desc() {
        return "lindit";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        model.get_param_tensors(tensors, prefix);
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                    struct ggml_tensor* timesteps,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* y,
                                    std::vector<int> skip_layers = std::vector<int>()) {
        struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, LINDIT_GRAPH_SIZE, false);

        x         = to_backend(x);
        context   = to_backend(context);
        y         = to_backend(y);
        timesteps = to_backend(timesteps);

        struct ggml_tensor* out = model.forward(compute_ctx,
                                                x,
                                                timesteps,
                                                y,
                                                context,
                                                skip_layers);

        ggml_build_forward_expand(gf, out);

        return gf;
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* y,
                 struct ggml_tensor** output     = NULL,
                 struct ggml_context* output_ctx = NULL,
                 std::vector<int> skip_layers    = std::vector<int>()) {
        // x: [N, in_channels, h, w]
        // timesteps: [N, ]
        // context: [N, max_position, hidden_size]([N, 154, 4096]) or [1, max_position, hidden_size]
        // y: [N, adm_in_channels] or [1, adm_in_channels]
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(x, timesteps, context, y, skip_layers);
        };

        GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
    }
};

#endif