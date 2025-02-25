#ifndef __LINDIT_HPP__
#define __LINDIT_HPP__

#include "ggml_extend.hpp"
#include "model.h"

namespace LinDiT {

    // TODO: maybe abstract these away? (they are the same as mmdit.h, just with sometimes different block names)
    struct PatchEmbed : public GGMLBlock {
        // 2D Image to Patch Embedding
    protected:
        bool flatten;
        bool dynamic_img_pad;
        int patch_size;

    public:
        PatchEmbed(int64_t img_size     = 224,
                   int patch_size       = 16,
                   int64_t in_chans     = 3,
                   int64_t embed_dim    = 1536,
                   bool bias            = true,
                   bool flatten         = true,
                   bool dynamic_img_pad = true)
            : patch_size(patch_size),
              flatten(flatten),
              dynamic_img_pad(dynamic_img_pad) {
            // img_size is always None
            // patch_size is always 2
            // in_chans is always 16
            // norm_layer is always False
            // strict_img_size is always true, but not used

            blocks["proj"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_chans,
                                                                   embed_dim,
                                                                   {patch_size, patch_size},
                                                                   {patch_size, patch_size},
                                                                   {0, 0},
                                                                   {1, 1},
                                                                   bias));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
            // x: [N, C, H, W]
            // return: [N, H*W, embed_dim]
            auto proj = std::dynamic_pointer_cast<Conv2d>(blocks["proj"]);

            if (dynamic_img_pad) {
                int64_t W = x->ne[0];
                int64_t H = x->ne[1];
                int pad_h = (patch_size - H % patch_size) % patch_size;
                int pad_w = (patch_size - W % patch_size) % patch_size;
                x         = ggml_pad(ctx, x, pad_w, pad_h, 0, 0);  // TODO: reflect pad mode
            }
            x = proj->forward(ctx, x);

            if (flatten) {
                x = ggml_reshape_3d(ctx, x, x->ne[0] * x->ne[1], x->ne[2], x->ne[3]);
                x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
            }
            return x;
        }
    };

    struct TimestepEmbedder : public GGMLBlock {
        // Embeds scalar timesteps into vector representations.
    protected:
        int64_t frequency_embedding_size;

    public:
        TimestepEmbedder(int64_t hidden_size,
                         int64_t frequency_embedding_size = 256)
            : frequency_embedding_size(frequency_embedding_size) {
            blocks["mlp.0"] = std::shared_ptr<GGMLBlock>(new Linear(frequency_embedding_size, hidden_size, true, true));
            blocks["mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size, true, true));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* t) {
            // t: [N, ]
            // return: [N, hidden_size]
            auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
            auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);

            auto t_freq = ggml_nn_timestep_embedding(ctx, t, frequency_embedding_size);  // [N, frequency_embedding_size]

            auto t_emb = mlp_0->forward(ctx, t_freq);
            t_emb      = ggml_silu_inplace(ctx, t_emb);
            t_emb      = mlp_2->forward(ctx, t_emb);
            return t_emb;
        }
    };

    struct CaptionProjection : public GGMLBlock {
    public:
        CaptionProjection(int64_t in_features,
                          int64_t out_features = -1,
                          bool bias            = true) {
            if (out_features == -1) {
                out_features = in_features;
            }
            blocks["linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(in_features, out_features, bias));
            blocks["linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(out_features, out_features, bias));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
            // x: [N, n_token, in_features]
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);

            x = linear_1->forward(ctx, x);
            // TODO: is it gelu?
            x = ggml_gelu_inplace(ctx, x);
            x = linear_2->forward(ctx, x);
            return x;
        }
    };

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
        int64_t offset = m->nb[1] * m->ne[1];
        std::vector<struct ggml_tensor*> msa{
            ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], 0),           // shift
            ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 1),  // scale
            ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 2)   // gate
        };
        std::vector<struct ggml_tensor*> ffn{
            ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 3),  // shift
            ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 4),  // scale
            ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 5)   // gate
        };
        return {{msa, ffn}};
    }
    struct LinearAttentionHead : public GGMLBlock {
    public:
        LinearAttentionHead(int64_t dim, bool qkv_bias = false) {
            blocks["to_q"]     = std::shared_ptr<GGMLBlock>(new Linear(dim, qkv_bias));
            blocks["to_k"]     = std::shared_ptr<GGMLBlock>(new Linear(dim, qkv_bias));
            blocks["to_v"]     = std::shared_ptr<GGMLBlock>(new Linear(dim, qkv_bias));
            blocks["to_out.0"] = std::shared_ptr<GGMLBlock>(new Linear(dim, true));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x_in, struct ggml_tensor* x, std::vector<struct ggml_tensor*> shifts_scales_gate) {
            // linear attention: out = relu(q)*(relu(transpose(k))*v))
            auto to_q   = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k   = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v   = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto to_out = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);
            // x:    [N, seq_len, hidden_size]
            // x_in: [N, seq_len, hidden_size] (=x for self attn)

            struct ggml_tensor* q = to_q->forward(ctx, x);     // [N, seq_len, dim]
            struct ggml_tensor* k = to_k->forward(ctx, x_in);  // [N, seq_len, dim]
            struct ggml_tensor* v = to_v->forward(ctx, x_in);  // [N, seq_len, dim]

            // TODO: is that the right OP?
            k = ggml_transpose(ctx, k);  // [N, dim, seq_len]

            q = ggml_relu_inplace(ctx, q);
            k = ggml_relu_inplace(ctx, k);

            struct ggml_tensor* kv  = ggml_mul_mat(ctx, k, v);   // [N, dim, dim]
            struct ggml_tensor* out = ggml_mul_mat(ctx, q, kv);  // [N, seq_len, dim]

            out = to_out->forward(ctx, out);  // [N, seq_len, hidden_size]

            out = modulate_gated(ctx, out, shifts_scales_gate[0], shifts_scales_gate[1], shifts_scales_gate[2]);
            return out;
        }
    };

    struct MixFFN : public GGMLBlock {
    public:
        MixFFN(int64_t hidden_size = 1152, int64_t ffn_dim = 2880) {
            blocks["conv_inverted"] = std::shared_ptr<GGMLBlock>(new Conv2d(hidden_size, ffn_dim * 2, {1, 1}));
            blocks["conv_depth"]    = std::shared_ptr<GGMLBlock>(new Conv2d(ffn_dim * 2, ffn_dim * 2, {3, 3}));
            blocks["conv_point"]    = std::shared_ptr<GGMLBlock>(new Conv2d(ffn_dim, hidden_size, {1, 1}, {1, 1}, {0, 0}, {1, 1}, false));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
            auto c_i = std::dynamic_pointer_cast<Conv2d>(blocks["conv_inverted"]);
            auto c_d = std::dynamic_pointer_cast<Conv2d>(blocks["conv_depth"]);
            auto c_p = std::dynamic_pointer_cast<Conv2d>(blocks["conv_point"]);

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
            blocks["attn1"] = std::shared_ptr<GGMLBlock>(new LinearAttentionHead(dim));
            blocks["attn2"] = std::shared_ptr<GGMLBlock>(new LinearAttentionHead(dim, true));

            blocks["ff"] = std::shared_ptr<GGMLBlock>(new MixFFN(dim));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* y, struct ggml_tensor* t) {
            auto self_attn  = std::dynamic_pointer_cast<LinearAttentionHead>(blocks["attn1"]);
            auto cross_attn = std::dynamic_pointer_cast<LinearAttentionHead>(blocks["attn2"]);
            auto ff         = std::dynamic_pointer_cast<MixFFN>(blocks["ff"]);

            std::array<std::vector<struct ggml_tensor*>, 2> modulations = chunk_table(ctx, params["scale_shift_table"]);  // {{shift_msa, scale_msa, gate_msa}, {shift_ffn, scale_ffn, gate_ffn}}

            struct ggml_tensor* context = ggml_concat(ctx, x, t, 2);  // [N, seq_len, hidden_size]

            x = self_attn->forward(ctx, x, x, modulations[0]);
            x = cross_attn->forward(ctx, x, y, modulations[1]);
            x = ff->forward(ctx, x);

            return x;
        }
    };
    struct TimeEmbed : public GGMLBlock {
    public:
        TimeEmbed(int64_t hidden_size, int64_t time_embed_dim) {
            // TODO: init
            blocks["emb"]    = std::shared_ptr<GGMLBlock>(new TimestepEmbedder(hidden_size));
            blocks["linear"] = std::shared_ptr<Linear>(new Linear(hidden_size, time_embed_dim, true));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* t) {
            auto emb    = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["emb"]);
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

            t = emb->forward(ctx, t);
            t = linear->forward(ctx, t);
            return t;
        }
    };

    struct LinDiT : public GGMLBlock {
    protected:
        int depth              = 28;
        int64_t hidden_size    = 1152;
        int64_t caption_dim    = 2304;
        int64_t time_embed_dim = 6912;
        int64_t in_channels    = 32;
        int64_t patch_size     = 1;

        void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, std::string prefix = "") {
            enum ggml_type caption_norm_wtype = (tensor_types.find(prefix + "caption_norm.weight") != tensor_types.end()) ? tensor_types[prefix + "caption_norm.weight"] : GGML_TYPE_F32;
            params["caption_norm.weight"]     = ggml_new_tensor_1d(ctx, caption_norm_wtype, hidden_size);

            enum ggml_type sst_wtype    = (tensor_types.find(prefix + "scale_shift_table") != tensor_types.end()) ? tensor_types[prefix + "scale_shift_table"] : GGML_TYPE_F32;
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, sst_wtype, hidden_size, 2);
        }

    public:
        LinDiT(std::map<std::string, enum ggml_type>& tensor_types) {
            blocks["patch_embed"]        = std::shared_ptr<GGMLBlock>(new PatchEmbed(-1, patch_size, in_channels, hidden_size, true));
            blocks["time_embed"]         = std::shared_ptr<GGMLBlock>(new TimeEmbed(hidden_size, time_embed_dim));
            blocks["caption_projection"] = std::shared_ptr<GGMLBlock>(new CaptionProjection(caption_dim, hidden_size));
            blocks["proj_out"]           = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, in_channels, true));

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
            return x;
        }
    };

    struct LinDiTRunner : public GGMLRunner {
        LinDiT model;
        static std::map<std::string, enum ggml_type> empty_tensor_types;

        // TODO: flash attention
        LinDiTRunner(ggml_backend_t backend,
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
}
#endif