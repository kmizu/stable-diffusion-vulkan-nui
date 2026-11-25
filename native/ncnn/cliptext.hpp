#pragma once

#include <algorithm>
#include <cstdio>
#include <fcntl.h>
#include <io.h>
#include <limits>
#include <string>
#include <net.h>
#include <gpu.h>

// One encoder stays on the selected adapter for the whole ranking operation. Each
// request is a LE int32 count, then that many token IDs. Reply is the EOT projection.
static int clip_text(int argc, wchar_t** argv) {
    if (argc != 5) return 2;
    const std::wstring wanted(argv[2]);
    int adapter = -1;
    for (int i = 0; i < ncnn::get_gpu_count(); ++i) {
        const std::string name(ncnn::get_gpu_info(i).device_name());
        if (wanted == std::wstring(name.begin(), name.end())) {
            if (adapter != -1) return 3;
            adapter = i;
        }
    }
    if (adapter == -1) return 3;
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    ncnn::Net net;
    net.set_vulkan_device(adapter);
    net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_arithmetic = false;
    if (net.load_param(argv[3]) || net.load_model(argv[4]) || !net.opt.use_vulkan_compute) return 4;
    int count;
    while (fread(&count, sizeof(int), 1, stdin) == 1) {
        if (count < 2 || count > 77) return 5;
        ncnn::Mat tokens(count), pos(count), mask(count, count);
        int* ids = static_cast<int*>(tokens.data);
        int* indices = static_cast<int*>(pos.data);
        if (fread(ids, sizeof(int), count, stdin) != static_cast<size_t>(count)) return 5;
        for (int t = 0; t < count; ++t) {
            if (ids[t] < 0 || ids[t] >= 49408) return 5;
            indices[t] = t;
            float* row = mask.row(t);
            for (int k = 0; k < count; ++k) row[k] = k <= t ? 0.0f : -std::numeric_limits<float>::infinity();
        }
        ncnn::Mat out;
        auto ex = net.create_extractor();
        if (ex.input("tokens", tokens) || ex.input("positions", pos) || ex.input("mask", mask) || ex.extract("out", out)) return 6;
        if (out.w != 1024 || out.h != count || out.c != 1 || out.elemsize != 4) return 7;
        if (fwrite(out.row(count - 1), sizeof(float), 1024, stdout) != 1024 || fflush(stdout)) return 8;
    }
    return feof(stdin) ? 0 : 5;
}
