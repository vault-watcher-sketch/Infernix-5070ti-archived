// infernix-probe: prints the wide-arena sizing grid (pure capacity math over the loaded artifact's
// config; no model materialize, no serving). One CSV row per configuration for arena analysis.
#include "artifact/reader.h"
#include "infernix/types.h"
#include "models/load_options.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/execution/dflash2.h"
#include "models/qwen4_exp/execution/forward.h"
#include "models/qwen4_exp/load.h"

#include <cstdio>
#include <string>
#include <vector>

namespace qx = infernix::models::qwen4_exp;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: infernix-probe MODEL.infernix DRAFT.infernix [max_context] [lanes]\n");
        return 2;
    }
    const std::string model_path = argv[1], draft_path = argv[2];
    setvbuf(stdout, nullptr, _IONBF, 0);
    unsigned mc_default       = argc > 3 ? static_cast<unsigned>(std::stoul(argv[3])) : 262144U;
    int lanes                 = argc > 4 ? std::stoi(argv[4]) : 2;

    infernix::artifact::Reader reader(model_path);
    infernix::models::LoadOptions lo;
    lo.speculative   = infernix::SpeculativeBackend::DFlash2;
    lo.draft_model   = draft_path;
    lo.proposal_head = infernix::ProposalHead::Optimized; // --lm-head-draft
    auto plan        = qx::plan_load(reader, lo);
    const qx::Config& cfg = plan.config();
    const qx::TextConfig& c = cfg.text;

    std::printf("# hidden=%u attn.heads=%u kv_heads=%u head_dim=%u q_w=%u k_w=%u\n", unsigned(c.hidden_size),
                unsigned(c.attention.heads), unsigned(c.attention.kv_heads), unsigned(c.attention.head_dim),
                unsigned(c.attention.query_width()), unsigned(c.attention.key_width()));
    std::printf("# qsa.index_heads=%u index_head_dim=%u budget=%u compress_ratio=%u max_pos=%u layers=%u attn_layers=%u\n",
                unsigned(c.qsa.index_heads), unsigned(c.qsa.index_head_dim), unsigned(c.qsa.budget),
                unsigned(c.qsa.compress_ratio), unsigned(c.max_position_embeddings), unsigned(c.num_hidden_layers),
                unsigned(c.attention_layers));
    if (cfg.dflash2) {
        const auto& d = *cfg.dflash2;
        std::printf("# dflash2: window=%u layers=%u I=%u conv_kernel=%u conv_group=%u rank=%u head_rows=%u fused=%d taps=%zu\n",
                    unsigned(d.window), unsigned(d.layers), unsigned(d.intermediate), unsigned(d.conv_kernel),
                    unsigned(d.conv_group), unsigned(d.selector_rank), unsigned(d.head_rows), int(d.fused_head),
                    d.target_layer_ids.size());
    } else {
        std::printf("# dflash2: ABSENT (draft companion not bound)\n");
    }

    const auto kv = infernix::KvCacheStorage::Int8Group64;
    std::printf("kind,columns,max_context,bytes\n");
    for (const int cols : {512, 1024, 2048, 4096, 8192, 16384}) {
        const std::vector<unsigned> mcs = mc_default > 65536U ? std::vector<unsigned>{mc_default}
                                                              : std::vector<unsigned>{65536U, 131072U, 262144U};
        for (const unsigned mc : mcs) {
            std::printf("forward,%d,%u,%zu\n", cols, mc,
                        qx::execution::Forward::workspace_bytes(c, cols, std::int32_t(mc), kv,
                                                                infernix::ops::offloaded_moe::WideA16::kExact));
        }
    }
    if (cfg.dflash2) {
        const int k = 7; // --draft-tokens auto on Qwen4Exp resolves to 7
        for (const int ctxcols : {512, 1024, 2048}) {
            std::printf("dflash2_ctxcols,%d,%u,%zu\n", ctxcols, mc_default,
                        qx::execution::DFlash2::workspace_bytes(c, *cfg.dflash2, lanes, k + 1, ctxcols, mc_default));
        }
        std::printf("dflash2_window_arg,%u,%u,%zu\n", unsigned(cfg.dflash2->window), mc_default,
                    qx::execution::DFlash2::workspace_bytes(c, *cfg.dflash2, lanes, k + 1,
                                                            std::int32_t(cfg.dflash2->window), mc_default));
    }
    return 0;
}
