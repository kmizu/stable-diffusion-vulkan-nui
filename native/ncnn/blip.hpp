#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <io.h>
#include <string>
#include <net.h>
#include <gpu.h>

static bool load_blip(ncnn::Net& net, int adapter, const wchar_t* param, const wchar_t* bin) {
    net.set_vulkan_device(adapter);
    net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_arithmetic = false;
    return !net.load_param(param) && !net.load_model(bin) && net.opt.use_vulkan_compute;
}

// argv: --blip device vision.param vision.bin decoder.param decoder.bin.
// The decoder evaluates a causal 64-token window. Future padding has no influence on the current position.
static int blip_caption(int argc, wchar_t** argv) {
    if (argc != 7 && argc != 8) return 2;
    const bool git = std::wstring(argv[1]) == L"--git";
    const int side = git ? 224 : 384;
    const int image_tokens = git ? 257 : 577;
    const int vocab = git ? 30522 : 30524;
    int limit = 60;
    try { if (argc == 8) limit = std::stoi(argv[7]); } catch (...) { return 2; }
    if (limit < 1 || limit > 64) return 2;
    const std::wstring wanted(argv[2]);
    int adapter = -1;
    for (int i = 0; i < ncnn::get_gpu_count(); ++i) {
        const std::string device(ncnn::get_gpu_info(i).device_name());
        if (wanted == std::wstring(device.begin(), device.end())) {
            if (adapter != -1) return 3;
            adapter = i;
        }
    }
    if (adapter == -1) return 3;
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    ncnn::Mat features;
    {
        ncnn::Net vision;
        if (!load_blip(vision, adapter, argv[3], argv[4])) return 4;
        ncnn::Mat image(side, side, 3);
        if (image.empty()) return 5;
        for (int ch = 0; ch < 3; ++ch) {
            if (fread(image.channel(ch).data, sizeof(float), static_cast<size_t>(side) * side, stdin) != static_cast<size_t>(side) * side) return 5;
        }
        if (fgetc(stdin) != EOF) return 5;
        auto ex = vision.create_extractor();
        ncnn::Mat encoded;
        if (ex.input("in0", image) || ex.extract("out0", encoded)) return 6;
        if ((encoded.w != 768 && encoded.w != 1024) || encoded.h != image_tokens || encoded.c != 1 || encoded.elemsize != 4) return 7;
        features = encoded.clone();
    }
    ncnn::Net decoder;
    if (!load_blip(decoder, adapter, argv[5], argv[6])) return 4;
    ncnn::Mat tokens(64);
    auto* token = reinterpret_cast<int*>(tokens.data);
    std::fill(token, token + 64, 0);
    token[0] = git ? 101 : 30522;
    for (int step = 0; step < limit; ++step) {
        auto ex = decoder.create_extractor();
        ncnn::Mat result;
        if (ex.input("in0", tokens) || ex.input("in1", features) || ex.extract("out0", result)) return 6;
        if (result.w != vocab || result.h != 64 || result.c != 1 || result.elemsize != 4) return 7;
        const float* logits = result.row(step);
        const int id = static_cast<int>(std::max_element(logits, logits + result.w) - logits);
        if (!std::isfinite(logits[id])) return 7;
        if (id == 102) break;
        if (step + 1 < 64) token[step + 1] = id;
        if (fwrite(&id, sizeof(id), 1, stdout) != 1) return 8;
    }
    return fflush(stdout) == 0 ? 0 : 8;
}
