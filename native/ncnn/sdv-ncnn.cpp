#include "codebook.hpp"
#include "blip.hpp"
#include "cliptext.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <io.h>
#include <iostream>
#include <limits>
#include <string>
#include <net.h>
#include <gpu.h>

// stdin and stdout carry planar, little-endian F32 only. Diagnostics go to stderr.
static int run(int argc, wchar_t** argv) {
    if (argc != 9 && argc != 11) {
        std::cerr << "sdv-ncnn adapter param bin input output width height channels [scalar_input value]\n";
        return 2;
    }
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    int w, h, c;
    try { w = std::stoi(argv[6]); h = std::stoi(argv[7]); c = std::stoi(argv[8]); }
    catch (...) { return 2; }
    if (w <= 0 || h <= 0 || c <= 0 || w > 16384 || h > 16384 || c > 16384 ||
        static_cast<int64_t>(w) * h * c > 64LL * 1024 * 1024) return 2;
    auto ascii = [](const wchar_t* s) {
        std::wstring wide(s);
        return std::string(wide.begin(), wide.end());
    };
    int adapter = -1;
    const auto wanted = ascii(argv[1]);
    for (int i = 0; i < ncnn::get_gpu_count(); ++i) {
        if (wanted == ncnn::get_gpu_info(i).device_name()) {
            if (adapter >= 0) { std::cerr << "Ambiguous Vulkan adapter name\n"; return 3; }
            adapter = i;
        }
    }
    if (adapter < 0) { std::cerr << "Selected Vulkan adapter unavailable\n"; return 3; }
    ncnn::Net net;
    net.register_custom_layer("SdvCodebookLookup", CodebookLookup_layer_creator);
    net.set_vulkan_device(adapter);
    net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_arithmetic = false;
    if (net.load_param(argv[2]) || net.load_model(argv[3])) return 4;
    if (!net.opt.use_vulkan_compute) { std::cerr << "Vulkan is unavailable for this model\n"; return 4; }
    ncnn::Mat input(w, h, c);
    if (input.empty()) return 5;
    size_t plane = static_cast<size_t>(w) * h;
    for (int i = 0; i < c; ++i)
        if (fread(input.channel(i).data, sizeof(float), plane, stdin) != plane) return 5;
    if (fgetc(stdin) != EOF) return 5;
    auto ex = net.create_extractor();
    if (ex.input(ascii(argv[4]).c_str(), input)) return 6;
    if (argc == 11) {
        ncnn::Mat value(1);
        try { value[0] = std::stof(argv[10]); } catch (...) { return 2; }
        if (ex.input(ascii(argv[9]).c_str(), value)) return 6;
    }
    ncnn::Mat result;
    if (ex.extract(ascii(argv[5]).c_str(), result) || result.empty() ||
        result.elemsize != sizeof(float) || result.elempack != 1) return 7;
    size_t block = static_cast<size_t>(result.w) * result.h * result.d;
    if (block * result.c > 64LL * 1024 * 1024) return 7;
    for (int i = 0; i < result.c; ++i)
        if (fwrite(result.channel(i).data, sizeof(float), block, stdout) != block) return 8;
    return fflush(stdout) == 0 ? 0 : 8;
}

int wmain(int argc, wchar_t** argv) {
    ncnn::create_gpu_instance();
    int status;
    if (argc > 1 && (std::wstring(argv[1]) == L"--blip" || std::wstring(argv[1]) == L"--git")) status = blip_caption(argc, argv);
    else if (argc > 1 && std::wstring(argv[1]) == L"--clip-text") status = clip_text(argc, argv);
    else status = run(argc, argv);
    ncnn::destroy_gpu_instance();
    return status;
}
