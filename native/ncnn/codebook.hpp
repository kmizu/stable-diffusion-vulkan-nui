#pragma once
#include "layer.h"
#include <algorithm>
#include <cstring>

// CodeFormer decodes one 1024-entry categorical distribution per latent position.
class CodebookLookup final : public ncnn::Layer {
public:
    int forward(const std::vector<ncnn::Mat>& bottom, std::vector<ncnn::Mat>& top,
                const ncnn::Option& opt) const override {
        if (bottom.size() != 2 || top.size() != 1) return -1;
        const ncnn::Mat& logits = bottom[0];
        const ncnn::Mat& table = bottom[1];
        if (logits.w != 1024 || logits.h != 256 || logits.c != 1 ||
            table.w != 256 || table.h != 1024 || table.c != 1 ||
            logits.elemsize != sizeof(float) || table.elemsize != sizeof(float)) return -1;
        top[0].create(256, 256, sizeof(float), opt.blob_allocator);
        if (top[0].empty()) return -100;
        for (int i = 0; i < logits.h; ++i) {
            const float* row = logits.row(i);
            const int best = static_cast<int>(std::max_element(row, row + 1024) - row);
            std::memcpy(top[0].row(i), table.row(best), 256 * sizeof(float));
        }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(CodebookLookup)
