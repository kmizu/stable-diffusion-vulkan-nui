package sdfx.vulkan;

import java.awt.image.BufferedImage;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.concurrent.CancellationException;

import static java.lang.foreign.ValueLayout.*;

/** Windows x64 binding for the f296cb0 stable-diffusion.cpp C API. */
public final class VulkanInference implements AutoCloseable {
    public static final class NativeFailure extends IllegalStateException {
        public final int status;

        private NativeFailure(int status, String message) {
            super("stable-diffusion.cpp (" + status + "): " + message);
            this.status = status;
        }
    }

    public record Device(String id, String name) {}
    public record Embedding(String name, String path) {}
    public record Lora(String path, float weight) {}
    public record Conditioning(List<Lora> loras, BufferedImage controlImage, double controlStrength, boolean vaeTiling,
                               boolean automaticTiling, boolean tileX, boolean tileY, String sampleArgs) {
        public Conditioning(List<Lora> loras, BufferedImage controlImage, double strength, boolean vaeTiling, boolean automaticTiling) {
            this(loras, controlImage, strength, vaeTiling, automaticTiling, false, false, "");
        }
        public static Conditioning empty() { return new Conditioning(List.of(), null, 1, false, true); }
    }
    public record ModelOptions(String weights, String placement, String vae, String tensorTypes,
                               String controlModel, List<Embedding> embeddings, String prediction, String tinyPreview) {
        public ModelOptions(String weights, String placement, String vae, String tensorTypes, String controlModel,
                List<Embedding> embeddings, String prediction) {
            this(weights, placement, vae, tensorTypes, controlModel, embeddings, prediction, "");
        }
        public ModelOptions withTinyPreview(String path) {
            return new ModelOptions(weights, placement, vae, tensorTypes, controlModel, embeddings, prediction, path);
        }
        public ModelOptions(String weights, String placement, String vae, String tensorTypes,
                            String controlModel, List<Embedding> embeddings) {
            this(weights, placement, vae, tensorTypes, controlModel, embeddings, "");
        }
    }
    public record Hires(String upscaler, String modelPath, int width, int height, int steps, double strength) {}
    public record ImageOptions(BufferedImage init, BufferedImage mask, double strength, Hires hires, double maskConditioningWeight) {
        public ImageOptions(BufferedImage init, BufferedImage mask, double strength, Hires hires) {
            this(init, mask, strength, hires, 1.0);
        }
    }
    public record Latent(int width, int height, float[] values) {
        public Latent {
            if (width <= 0 || height <= 0 || (long) width * height * 4 != values.length || values.length > 16777216)
                throw new IllegalArgumentException("Invalid four-channel VAE latent dimensions");
        }
    }
    @FunctionalInterface public interface Progress {
        void update(String stage, long completed, long total);
    }

    @FunctionalInterface public interface Preview {
        void update(int step, int pass, int passSteps, boolean noisy, BufferedImage image);
    }

    private Double eta;
    private static final FunctionDescriptor PREVIEW = FunctionDescriptor.ofVoid(
        JAVA_INT, JAVA_INT, ADDRESS, JAVA_BOOLEAN, ADDRESS);
    private static final MethodHandle PREVIEW_HANDLER;
    private static final Linker LINKER = Linker.nativeLinker();
    private static final MemorySegment NULL = MemorySegment.NULL;
    private static final FunctionDescriptor PROGRESS = FunctionDescriptor.ofVoid(ADDRESS, ADDRESS);
    private static final FunctionDescriptor LOG = FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, ADDRESS);
    private static final MethodHandle EVENT_HANDLER;
    private static final MemorySegment LOG_STUB;
    private static final MemorySegment QUIET_PROGRESS;

    static {
        try {
            var lookup = MethodHandles.lookup();
            PREVIEW_HANDLER = lookup.findVirtual(VulkanInference.class, "onPreview",
                MethodType.methodType(void.class, Preview.class, int.class, int.class,
                    MemorySegment.class, boolean.class, MemorySegment.class));
            EVENT_HANDLER = lookup.findStatic(VulkanInference.class, "onProgress",
                MethodType.methodType(void.class, Progress.class, MemorySegment.class, MemorySegment.class));
            var log = lookup.findStatic(VulkanInference.class, "onLog",
                MethodType.methodType(void.class, int.class, MemorySegment.class, MemorySegment.class));
            // Both loader and sampler print bars to stdout when the global callback is absent.
            var bar = FunctionDescriptor.ofVoid(JAVA_INT, JAVA_INT, JAVA_FLOAT, ADDRESS);
            QUIET_PROGRESS = LINKER.upcallStub(MethodHandles.empty(bar.toMethodType()), bar, Arena.global());
            LOG_STUB = LINKER.upcallStub(log, LOG, Arena.global());
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private final Arena library = Arena.ofShared();
    private final SymbolLookup symbols;
    private final java.util.Map<String, MethodHandle> calls = new java.util.HashMap<>();
    private final Object lifetime = new Object();
    private MemorySegment ctx = NULL;
    private MemorySegment visionCtx = NULL;
    private boolean visionRunning;
    private boolean closed;
    private volatile boolean runningDgl;

    public VulkanInference(Path nativeDir) {
        Path dir = nativeDir.toAbsolutePath().normalize();
        try {
            for (String name : List.of("ggml-base", "ggml", "libsharpyuv", "libwebp", "libwebpdemux", "libwebpmux", "webm"))
                SymbolLookup.libraryLookup(dir.resolve(name + ".dll"), library);
            var ggml = SymbolLookup.libraryLookup(dir.resolve("ggml.dll"), library);
            var base = SymbolLookup.libraryLookup(dir.resolve("ggml-base.dll"), library);
            calls.put("ggml_backend_dev_by_name", LINKER.downcallHandle(ggml.findOrThrow("ggml_backend_dev_by_name"),
                FunctionDescriptor.of(ADDRESS, ADDRESS)));
            calls.put("ggml_backend_dev_memory", LINKER.downcallHandle(base.findOrThrow("ggml_backend_dev_memory"),
                FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS)));
            symbols = SymbolLookup.libraryLookup(dir.resolve("stable-diffusion.dll"), library);
            bind("sd_set_log_callback", null, ADDRESS, ADDRESS);
            call("sd_set_log_callback", LOG_STUB, NULL);
            bind("sd_set_progress_callback", null, ADDRESS, ADDRESS);
            call("sd_set_progress_callback", QUIET_PROGRESS, NULL);
            try (var arena = Arena.ofConfined()) {
                invoke(LINKER.downcallHandle(ggml.findOrThrow("ggml_backend_load_all_from_path"),
                    FunctionDescriptor.ofVoid(ADDRESS)), arena.allocateFrom(dir.toString()));
            }
            bind("sd_commit", ADDRESS);
            String commit = cString(ptr("sd_commit"));
            if (!commit.startsWith("f296cb0"))
                throw new IllegalStateException("Unsupported stable-diffusion.cpp DLL commit: " + commit);
            bind("sd_list_devices", JAVA_LONG, ADDRESS, JAVA_LONG);
            bind("sd_ctx_opt_create", ADDRESS);
            bind("sd_img_gen_opt_create", ADDRESS);
            bind("sd_opt_free", null, ADDRESS);
            bind("sd_opt_set_string", JAVA_INT, ADDRESS, ADDRESS, ADDRESS);
            bind("sd_opt_set_embeddings", JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG);
            bind("sd_opt_set_loras", JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG);
            bind("sd_opt_set_bool", JAVA_INT, ADDRESS, ADDRESS, JAVA_BOOLEAN);
            bind("sd_opt_set_int", JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG);
            bind("sd_opt_set_float", JAVA_INT, ADDRESS, ADDRESS, JAVA_DOUBLE);
            bind("sd_upscaler_opt_create", ADDRESS);
            bind("sd_upscale_opt_create", ADDRESS);
            bind("sd_upscaler_create", JAVA_INT, ADDRESS, ADDRESS);
            bind("sd_upscaler_free", null, ADDRESS);
            bind("sd_upscaler_get_scale", JAVA_INT, ADDRESS);
            bind("sd_upscale", JAVA_INT, ADDRESS, ADDRESS, ADDRESS);
            bind("sd_ctx_create", JAVA_INT, ADDRESS, ADDRESS);
            bind("sd_ctx_supports_image_generation", JAVA_BOOLEAN, ADDRESS);
            bind("sd_get_model_version_name", ADDRESS, ADDRESS);
            bind("sd_ctx_free", null, ADDRESS);
            bind("sd_generate_image", JAVA_INT, ADDRESS, ADDRESS, ADDRESS);
            bind("sd_set_generation_progress_callback", null, ADDRESS, ADDRESS, ADDRESS);
            bind("sd_get_preview_info", MemoryLayout.structLayout(JAVA_INT, JAVA_INT));
            bind("sd_set_preview_callback", JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, JAVA_BOOLEAN, JAVA_BOOLEAN, ADDRESS);
            bind("sd_cancel_generation", null, ADDRESS, JAVA_INT);
            bind("sd_result_free", null, ADDRESS);
            bind("sd_ctx_get_dgl_ctx", ADDRESS, ADDRESS);
            bind("sd_dgl_ctx_create", ADDRESS);
            bind("sd_dgl_ctx_free", null, ADDRESS);
            bind("sd_dgl_run", JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS);
            bind("sd_dgl_ctx_cancel", JAVA_INT, ADDRESS);
            bind("sd_dgl_ctx_set_progress_callback", JAVA_INT, ADDRESS, ADDRESS, ADDRESS);
            bind("sd_dgl_result_free", null, ADDRESS);
            bind("sd_dgl_result_tensor", ADDRESS, ADDRESS, ADDRESS);
            bind("sd_dgl_error", ADDRESS);
            bind("sd_dgl_builtin_catalog", ADDRESS);
            bind("sd_result_get_image", ADDRESS, ADDRESS, JAVA_LONG);
            bind("sd_tensor_dim", JAVA_LONG, ADDRESS, JAVA_LONG);
            bind("sd_tensor_byte_size", JAVA_LONG, ADDRESS);
            bind("sd_tensor_type", JAVA_INT, ADDRESS);
            bind("sd_tensor_rank", JAVA_LONG, ADDRESS);
            bind("sd_tensor_data", ADDRESS, ADDRESS);
            bind("sd_tensor_create", ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS, JAVA_LONG);
            bind("sd_tensor_free", null, ADDRESS);
            bind("sd_opt_set_tensor", JAVA_INT, ADDRESS, ADDRESS, ADDRESS);
            bind("sd_get_last_error", ADDRESS);
        } catch (RuntimeException | Error e) {
            library.close();
            throw e;
        }
    }

    private static void onLog(int level, MemorySegment message, MemorySegment unused) {
        try {
            if (level >= 2) System.err.println(cString(message).stripTrailing());
        } catch (Throwable ignored) { /* Exceptions must never cross native callbacks. */ }
    }

    private static void onProgress(Progress sink, MemorySegment event, MemorySegment unused) {
        try {
            // x64 sd_generation_progress_t: three pointers, int, padding, two int64_t.
            var p = event.reinterpret(48);
            sink.update(cString(p.get(ADDRESS, 0)), p.get(JAVA_LONG, 32), p.get(JAVA_LONG, 40));
        } catch (Throwable ignored) { /* Exceptions must never cross native callbacks. */ }
    }

    private void onPreview(Preview sink, int step, int count, MemorySegment frames, boolean noisy, MemorySegment unused) {
        try {
            if (count > 0 && !frames.equals(NULL)) {
                try (var arena = Arena.ofConfined()) {
                    var info = (MemorySegment) call("sd_get_preview_info", arena);
                    sink.update(step, info.get(JAVA_INT, 0), info.get(JAVA_INT, 4), noisy,
                        copyImage(frames.reinterpret(ADDRESS.byteSize()).get(ADDRESS, 0)));
                }
            }
        } catch (Throwable e) {
            System.err.println("Skipping preview: " + e.getMessage());
        }
    }

    // The worker serializes jobs. The upstream preview callback is process-wide.
    public AutoCloseable generationSettings(int interval, String mode, Double sampleEta, boolean denoised, boolean noisy, Preview sink) {
        var arena = Arena.ofShared();
        try {
            var disabled = arena.allocateFrom("none");
            boolean enabled = interval > 0 && !mode.equals("none");
            if (enabled) {
                var fn = LINKER.upcallStub(PREVIEW_HANDLER.bindTo(this).bindTo(sink), PREVIEW, arena);
                check(call("sd_set_preview_callback", fn, arena.allocateFrom(mode), interval, denoised, noisy, NULL));
            }
            eta = sampleEta;
            return () -> {
                eta = null;
                if (enabled) check(call("sd_set_preview_callback", NULL, disabled, 1, false, false, NULL));
                arena.close();
            };
        } catch (RuntimeException | Error e) {
            arena.close();
            throw e;
        }
    }

    private static String cString(MemorySegment address) {
        return address.equals(NULL) ? "" : address.reinterpret(Long.MAX_VALUE).getString(0);
    }

    private void bind(String name, MemoryLayout result, MemoryLayout... args) {
        var type = result == null ? FunctionDescriptor.ofVoid(args) : FunctionDescriptor.of(result, args);
        calls.put(name, LINKER.downcallHandle(symbols.findOrThrow(name), type));
    }

    private static Object invoke(MethodHandle fn, Object... args) {
        try {
            return fn.invokeWithArguments(args);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable e) {
            throw new IllegalStateException(e);
        }
    }

    private Object call(String name, Object... args) { return invoke(calls.get(name), args); }
    private MemorySegment ptr(String name, Object... args) { return (MemorySegment) call(name, args); }

    private void check(Object status) {
        int code = (int) status;
        if (code == 10) throw new CancellationException("Generation cancelled");
        if (code != 0) throw new NativeFailure(code, cString(ptr("sd_get_last_error")));
    }

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("Vulkan inference engine is closed");
    }

    public synchronized List<Device> devices() {
        ensureOpen();
        try (var arena = Arena.ofConfined()) {
            long size = (long) call("sd_list_devices", NULL, 0L) + 1;
            var buf = arena.allocate(size);
            call("sd_list_devices", buf, size);
            var devices = new ArrayList<Device>();
            for (String line : buf.getString(0).split("\\R")) {
                String[] fields = line.split("\\t", 2);
                if (fields.length == 2 && fields[0].matches("(?i)Vulkan[0-9]+"))
                    devices.add(new Device(fields[0], fields[1].strip()));
            }
            return List.copyOf(devices);
        }
    }

    public record DeviceMemory(long free, long total) {}

    public DeviceMemory memory(String adapter) {
        try (var arena = Arena.ofConfined()) {
            var dev = ptr("ggml_backend_dev_by_name", arena.allocateFrom(adapter));
            if (dev.equals(NULL)) throw new IllegalArgumentException("Unknown Vulkan device: " + adapter);
            var free = arena.allocate(JAVA_LONG);
            var total = arena.allocate(JAVA_LONG);
            call("ggml_backend_dev_memory", dev, free, total);
            return new DeviceMemory(free.get(JAVA_LONG, 0), total.get(JAVA_LONG, 0));
        }
    }

    private void string(Arena a, MemorySegment opt, String key, String value) {
        if (value.indexOf('\0') >= 0) throw new IllegalArgumentException("Embedded NUL in " + key);
        check(call("sd_opt_set_string", opt, a.allocateFrom(key), a.allocateFrom(value)));
    }

    private void integer(Arena a, MemorySegment opt, String key, long value) {
        check(call("sd_opt_set_int", opt, a.allocateFrom(key), value));
    }

    public void load(Path model, String deviceId) {
        load(model, deviceId, "");
    }

    public synchronized void load(Path model, String deviceId, String weightType) {
        load(model, deviceId, new ModelOptions(weightType, deviceId, "", "", "", List.of()));
    }

    public synchronized void load(Path model, String deviceId, ModelOptions options) {
        ensureOpen();
        if (!Files.isRegularFile(model)) throw new IllegalArgumentException("Missing model file: " + model);
        if (devices().stream().noneMatch(d -> d.id().equals(deviceId)))
            throw new IllegalArgumentException("Unknown Vulkan device: " + deviceId);
        try (var a = Arena.ofConfined()) {
            var opt = newOptions("sd_ctx_opt_create");
            try {
                string(a, opt, "model_path", model.toAbsolutePath().normalize().toString());
                string(a, opt, "backend", deviceId);
                if (!options.prediction().isEmpty()) string(a, opt, "prediction", options.prediction());
                if (!options.weights().isEmpty()) string(a, opt, "wtype", options.weights());
                if (!options.placement().isEmpty()) string(a, opt, "params_backend", options.placement());
                if (!options.tensorTypes().isEmpty()) string(a, opt, "tensor_type_rules", options.tensorTypes());
                if (!options.vae().isEmpty()) {
                    string(a, opt, "vae_path", options.vae());
                    check(call("sd_opt_set_bool", opt, a.allocateFrom("force_sdxl_vae_conv_scale"), true));
                }
                if (!options.tinyPreview().isEmpty()) {
                    string(a, opt, "taesd_path", options.tinyPreview());
                    check(call("sd_opt_set_bool", opt, a.allocateFrom("tae_preview_only"), true));
                }
                if (!options.controlModel().isEmpty()) string(a, opt, "control_net_path", options.controlModel());
                if (!options.embeddings().isEmpty()) {
                    // x64 sd_embedding_t is two consecutive pointers.
                    var entries = a.allocate(16L * options.embeddings().size(), 8);
                    for (int i = 0; i < options.embeddings().size(); i++) {
                        var entry = options.embeddings().get(i);
                        entries.set(ADDRESS, 16L*i, a.allocateFrom(entry.name()));
                        entries.set(ADDRESS, 16L*i+8, a.allocateFrom(entry.path()));
                    }
                    check(call("sd_opt_set_embeddings", opt, a.allocateFrom("embeddings"), entries, (long) options.embeddings().size()));
                }
                check(call("sd_opt_set_bool", opt, a.allocateFrom("enable_mmap"), true));
                var output = a.allocate(ADDRESS);
                check(call("sd_ctx_create", opt, output));
                var candidate = output.get(ADDRESS, 0);
                if (!(boolean) call("sd_ctx_supports_image_generation", candidate)) {
                    String error = cString(ptr("sd_get_last_error"));
                    call("sd_ctx_free", candidate);
                    throw new IllegalArgumentException("Unsupported image model: " + model + " " + error);
                }
                synchronized (lifetime) {
                    if (!ctx.equals(NULL)) call("sd_ctx_free", ctx);
                    ctx = candidate;
                }
            } finally { call("sd_opt_free", opt); }
        }
    }

    private MemorySegment newOptions(String fn) {
        var result = ptr(fn);
        if (result.equals(NULL)) throw new IllegalStateException("Cannot create native options: " + cString(ptr("sd_get_last_error")));
        return result;
    }

    private void applyConditioning(Arena a, MemorySegment opt, Conditioning inputs) {
        if (!inputs.sampleArgs().isEmpty()) string(a, opt, "sample.args", inputs.sampleArgs());
        if (inputs.tileX()) check(call("sd_opt_set_bool", opt, a.allocateFrom("circular_x"), true));
        if (inputs.tileY()) check(call("sd_opt_set_bool", opt, a.allocateFrom("circular_y"), true));
        if (inputs.vaeTiling())
            check(call("sd_opt_set_bool", opt, a.allocateFrom("vae_tiling.enabled"), true));
        if (!inputs.loras().isEmpty()) {
            // x64 sd_lora_t: bool, three padding bytes, float, pointer.
            var entries = a.allocate(16L * inputs.loras().size(), 8);
            for (int i = 0; i < inputs.loras().size(); i++) {
                var lora = inputs.loras().get(i);
                entries.set(JAVA_FLOAT, 16L*i+4, lora.weight());
                entries.set(ADDRESS, 16L*i+8, a.allocateFrom(lora.path()));
            }
            check(call("sd_opt_set_loras", opt, a.allocateFrom("loras"), entries, (long) inputs.loras().size()));
        }
        if (inputs.controlImage() != null) {
            imageTensor(a, opt, "control_image", inputs.controlImage(), 3);
            check(call("sd_opt_set_float", opt, a.allocateFrom("control_strength"), inputs.controlStrength()));
        }
    }

    public synchronized String modelVersion() {
        ensureOpen();
        if (ctx.equals(NULL)) throw new IllegalStateException("No model loaded");
        return cString(ptr("sd_get_model_version_name", ctx));
    }

    public BufferedImage generate(String prompt, String negativePrompt, int width, int height, int steps,
            double cfg, long seed, String sampler, String scheduler, Progress progress) {
        return generate(prompt, negativePrompt, width, height, steps, cfg, seed, sampler, scheduler, 1, progress);
    }

    public BufferedImage generate(String prompt, String negativePrompt, int width, int height, int steps,
            double cfg, long seed, String sampler, String scheduler, int clipSkip, Progress progress) {
        return generate(prompt, negativePrompt, width, height, steps, cfg, seed, sampler, scheduler,
                clipSkip, null, Conditioning.empty(), progress);
    }

    public synchronized BufferedImage generate(String prompt, String negativePrompt, int width, int height, int steps,
            double cfg, long seed, String sampler, String scheduler, int clipSkip, ImageOptions image,
            Conditioning conditioning, Progress progress) {
        ensureOpen();
        if (ctx.equals(NULL)) throw new IllegalStateException("No model loaded");
        try (var a = Arena.ofShared()) {
            var opt = newOptions("sd_img_gen_opt_create");
            try {
                string(a, opt, "prompt", prompt);
                string(a, opt, "negative_prompt", negativePrompt);
                integer(a, opt, "width", width);
                integer(a, opt, "height", height);
                integer(a, opt, "sample.steps", steps);
                if (eta != null) check(call("sd_opt_set_float", opt, a.allocateFrom("sample.eta"), eta));
                integer(a, opt, "clip_skip", clipSkip);
                integer(a, opt, "seed", seed);
                if (image != null) {
                    if (image.init != null) {
                        imageTensor(a, opt, "init_image", image.init, 3);
                        check(call("sd_opt_set_float", opt, a.allocateFrom("strength"), image.strength));
                    }
                    if (image.mask != null) {
                        imageTensor(a, opt, "mask_image", image.mask, 1);
                        if (image.maskConditioningWeight != 1.0)
                            check(call("sd_opt_set_float", opt, a.allocateFrom("mask_conditioning_weight"), image.maskConditioningWeight));
                    }
                    if (image.hires != null) {
                        Hires hires = image.hires;
                        check(call("sd_opt_set_bool", opt, a.allocateFrom("hires.enabled"), true));
                        string(a, opt, "hires.upscaler", hires.upscaler);
                        if (hires.modelPath != null) string(a, opt, "hires.model_path", hires.modelPath);
                        integer(a, opt, "hires.target_width", hires.width);
                        integer(a, opt, "hires.target_height", hires.height);
                        integer(a, opt, "hires.steps", hires.steps);
                        check(call("sd_opt_set_float", opt, a.allocateFrom("hires.denoising_strength"), hires.strength));
                    }
                }
                applyConditioning(a, opt, conditioning);
                check(call("sd_opt_set_float", opt, a.allocateFrom("sample.guidance.txt_cfg"), cfg));
                string(a, opt, "sample.method", sampler);
                string(a, opt, "sample.scheduler", scheduler);
                var result = a.allocate(ADDRESS);
                var callback = progress == null ? NULL : LINKER.upcallStub(EVENT_HANDLER.bindTo(progress), PROGRESS, a);
                call("sd_set_generation_progress_callback", ctx, callback, NULL);
                try {
                    int status = (int) call("sd_generate_image", ctx, opt, result);
                    if (status == 6 && conditioning.automaticTiling() && !conditioning.vaeTiling()) {
                        check(call("sd_opt_set_bool", opt, a.allocateFrom("vae_tiling.enabled"), true));
                        status = (int) call("sd_generate_image", ctx, opt, result);
                    }
                    check(status);
                    var handle = result.get(ADDRESS, 0);
                    try {
                        return copyImage(ptr("sd_result_get_image", handle, 0L));
                    } finally { call("sd_result_free", handle); }
                } finally { call("sd_set_generation_progress_callback", ctx, NULL, NULL); }
            } finally { call("sd_opt_free", opt); }
        }
    }

    public synchronized BufferedImage upscale(Path path, String deviceId, BufferedImage image, int width, int height) {
        ensureOpen();
        try (var a = Arena.ofConfined()) {
            var model = newOptions("sd_upscaler_opt_create");
            var out = a.allocate(ADDRESS);
            try {
                string(a, model, "model_path", path.toAbsolutePath().toString());
                string(a, model, "backend", deviceId);
                check(call("sd_upscaler_create", model, out));
            } finally { call("sd_opt_free", model); }
            var handle = out.get(ADDRESS, 0);
            try {
                int scale = (int) call("sd_upscaler_get_scale", handle);
                if (scale < 1) throw new IllegalArgumentException("Invalid model upscaler scale");
                int repeats = 0;
                long w = image.getWidth(), h = image.getHeight();
                do {
                    w = Math.multiplyExact(w, scale);
                    h = Math.multiplyExact(h, scale);
                    if (w > 16384 || h > 16384 || w * h > 16777216L)
                        throw new IllegalArgumentException("Model upscale intermediate exceeds pixel limit");
                    ++repeats;
                } while (scale > 1 && (w < width || h < height) && repeats < 3);
                var opt = newOptions("sd_upscale_opt_create");
                try {
                    imageTensor(a, opt, "image", image, 3);
                    integer(a, opt, "repeats", repeats);
                    check(call("sd_upscale", handle, opt, out));
                    var result = out.get(ADDRESS, 0);
                    try { return copyImage(ptr("sd_result_get_image", result, 0L)); }
                    finally { call("sd_result_free", result); }
                } finally { call("sd_opt_free", opt); }
            } finally { call("sd_upscaler_free", handle); }
        }
    }

    private MemorySegment createImageTensor(Arena a, BufferedImage image, int channels) {
        int width = image.getWidth(), height = image.getHeight();
        var pixels = image.getRGB(0, 0, width, height, null, 0, width);
        byte[] bytes = new byte[Math.multiplyExact(pixels.length, channels)];
        int n = 0;
        for (int pixel : pixels) {
            if (channels == 3) {
                bytes[n++] = (byte) (pixel >> 16);
                bytes[n++] = (byte) (pixel >> 8);
            }
            bytes[n++] = (byte) pixel;
        }
        var shape = a.allocateFrom(JAVA_LONG, channels, width, height);
        var tensor = ptr("sd_tensor_create", 1, 3L, shape, a.allocateFrom(JAVA_BYTE, bytes), (long) bytes.length);
        if (tensor.equals(NULL)) throw new IllegalArgumentException(cString(ptr("sd_get_last_error")));
        return tensor;
    }

    private void imageTensor(Arena a, MemorySegment opt, String key, BufferedImage image, int channels) {
        var tensor = createImageTensor(a, image, channels);
        try {
            check(call("sd_opt_set_tensor", opt, a.allocateFrom(key), tensor));
        } finally { call("sd_tensor_free", tensor); }
    }

    private <T> T dglProgram(Arena arena, String source, Map<String, MemorySegment> inputs,
            String resultName, Function<MemorySegment,T> copy) {
        if (!standaloneDgl.equals(NULL)) return dglRun(standaloneDgl, arena, source, inputs, resultName, copy);
        if (ctx.equals(NULL)) throw new IllegalStateException("Load a checkpoint before using DGL");
        var dgl = ptr("sd_ctx_get_dgl_ctx", ctx);
        return dglRun(dgl, arena, source, inputs, resultName, copy);
    }

    private <T> T dglRun(MemorySegment dgl, Arena arena, String source, Map<String, MemorySegment> inputs,
            String resultName, Function<MemorySegment,T> copy) {
        var args = inputs.isEmpty() ? NULL : arena.allocate(16L * inputs.size(), 8);
        int offset = 0;
        for (var entry : inputs.entrySet()) {
            args.set(ADDRESS, offset, arena.allocateFrom(entry.getKey()));
            args.set(ADDRESS, offset + 8, entry.getValue());
            offset += 16;
        }
        var output = arena.allocate(ADDRESS);
        if (dglProgress != null) {
            var callback = LINKER.upcallStub(EVENT_HANDLER.bindTo(dglProgress), PROGRESS, arena);
            check(call("sd_dgl_ctx_set_progress_callback", dgl, callback, NULL));
        }
        runningDgl = true;
        int status;
        try {
            status = (int) call("sd_dgl_run", dgl, arena.allocateFrom(source), args, (long) inputs.size(), output);
        } finally {
            runningDgl = false;
            if (dglProgress != null) call("sd_dgl_ctx_set_progress_callback", dgl, NULL, NULL);
        }
        if (status == 10) throw new CancellationException("DGL operation interrupted");
        if (status != 0) throw new NativeFailure(status, cString(ptr("sd_dgl_error")));
        var result = output.get(ADDRESS, 0);
        try {
            var tensor = ptr("sd_dgl_result_tensor", result, arena.allocateFrom(resultName));
            if (tensor.equals(NULL)) throw new IllegalStateException("DGL output tensor missing: " + resultName);
            return copy.apply(tensor);
        } finally { call("sd_dgl_result_free", result); }
    }

    private <T> T dglWithImages(String source, Map<String,BufferedImage> images, String resultName,
            Map<String,Latent> noises, Function<MemorySegment,T> copy) {
        ensureOpen();
        try (var arena = Arena.ofConfined()) {
            var tensors = new java.util.LinkedHashMap<String,MemorySegment>();
            try {
                for (var entry : images.entrySet())
                    tensors.put(entry.getKey(), createImageTensor(arena, entry.getValue(), 3));
                for (var entry : noises.entrySet()) {
                    var noise = entry.getValue();
                    var shape = arena.allocateFrom(JAVA_LONG,noise.width,noise.height,4,1);
                    var tensor = ptr("sd_tensor_create",4,4L,shape,arena.allocateFrom(JAVA_FLOAT,noise.values),4L*noise.values.length);
                    if (tensor.equals(NULL)) throw new IllegalArgumentException(cString(ptr("sd_get_last_error")));
                    tensors.put(entry.getKey(),tensor);
                }
                return dglProgram(arena, source, tensors, resultName, copy);
            } finally { tensors.values().forEach(t -> call("sd_tensor_free", t)); }
        }
    }

    public synchronized BufferedImage dglImage(String program, Map<String,BufferedImage> inputs, String resultName) {
        return dglWithImages(program, inputs, resultName, Map.of(), this::copyImage);
    }

    public synchronized BufferedImage dglImage(String program, Map<String,BufferedImage> inputs, String resultName, Map<String,Latent> noises) {
        return dglWithImages(program, inputs, resultName, noises, this::copyImage);
    }

    public synchronized void writeDglTensors(String program) {
        ensureOpen();
        try (var arena = Arena.ofConfined()) {
            dglProgram(arena, program, Map.of(), "value", tensor -> null);
        }
    }

    public synchronized float[] detectBoxes(String source, BufferedImage image) {
        return dglWithImages(source, Map.of("input", image), "boxes", Map.of(), tensor -> {
            long size = (long)call("sd_tensor_byte_size",tensor);
            if ((int)call("sd_tensor_type",tensor) != 4 || size < 0 || size > 65536 || size % 16 != 0)
                throw new IllegalStateException("Expected floating YOLO boxes of shape [4,N]");
            return size == 0 ? new float[0] : ptr("sd_tensor_data",tensor).reinterpret(size).toArray(JAVA_FLOAT);
        });
    }

    public synchronized float[] clipVision(String program, BufferedImage image, boolean keepLoaded) {
        return clipVision(program, image, keepLoaded, 1024);
    }

    public synchronized float[] clipVision(String program, BufferedImage image, boolean keepLoaded, int dimensions) {
        ensureOpen();
        if (dimensions != 512 && dimensions != 768 && dimensions != 1024 && dimensions != 1280)
            throw new IllegalArgumentException("Unsupported CLIP projection size");
        long byteSize = 4L * dimensions;
        synchronized (lifetime) {
            if (visionCtx.equals(NULL)) visionCtx = ptr("sd_dgl_ctx_create");
            if (visionCtx.equals(NULL)) throw new IllegalStateException("Cannot create CLIP vision context");
            visionRunning = true;
        }
        try (var a = Arena.ofConfined()) {
            var input = createImageTensor(a, image, 3);
            try {
                return dglRun(visionCtx, a, program, Map.of("input",input), "embedding", tensor -> {
                    if ((int)call("sd_tensor_type",tensor) != 4 || (long)call("sd_tensor_byte_size",tensor) != byteSize)
                        throw new IllegalStateException("Unexpected CLIP projection dimension, expected " + dimensions);
                    return ptr("sd_tensor_data",tensor).reinterpret(byteSize).toArray(JAVA_FLOAT);
                });
            } finally { call("sd_tensor_free",input); }
        } finally {
            synchronized (lifetime) { visionRunning = false; }
            if (!keepLoaded) unloadClipVision();
        }
    }

    public synchronized boolean unloadClipVision() {
        synchronized (lifetime) {
            if (visionCtx.equals(NULL)) return false;
            call("sd_dgl_ctx_free",visionCtx);
            visionCtx = NULL;
            return true;
        }
    }

    public boolean supportsUnetResize() {
        return symbols.find("sd_nui_unet_resize_supported").isPresent();
    }

    public boolean supportsTiledDiffusion() {
        return cString(ptr("sd_dgl_builtin_catalog")).contains("`diffusion_tile_width?");
    }

    public boolean supportsLatentModifier() {
        return cString(ptr("sd_dgl_builtin_catalog")).contains("`latent_modifier?");
    }

    public boolean supportsNegPip() {
        return cString(ptr("sd_dgl_builtin_catalog")).contains("`clip_negpip?");
    }

    public boolean supportsDynamicThreshold() {
        return cString(ptr("sd_dgl_builtin_catalog")).contains("| `dynthresh_scale` |");
    }

    public boolean supportsClipIoEncodingOptions() {
        return cString(ptr("sd_dgl_builtin_catalog")).contains("`clip_no_special_tokens?");
    }

    public boolean supportsClipIoTokenStrings() {
        return cString(ptr("sd_dgl_builtin_catalog")).contains("`clip_return_tokens?");
    }

    public boolean supportsUnetHooks() {
        return symbols.find("sd_nui_unet_hooks_supported").isPresent();
    }

    public boolean supportsSvd() {
        return symbols.find("sd_nui_svd_supported").isPresent();
    }

    private Progress dglProgress;
    private MemorySegment standaloneDgl = NULL;

    public synchronized List<BufferedImage> dglFrames(String program, Map<String,BufferedImage> inputs, String resultName, Progress progress) {
        ensureOpen();
        synchronized (lifetime) {
            standaloneDgl = ptr("sd_dgl_ctx_create");
            if (standaloneDgl.equals(NULL)) throw new IllegalStateException("Cannot create SVD context");
        }
        dglProgress = progress;
        try {
            return dglFrames(program, inputs, resultName);
        } finally {
            dglProgress = null;
            synchronized (lifetime) {
                call("sd_dgl_ctx_free", standaloneDgl);
                standaloneDgl = NULL;
            }
        }
    }

    public synchronized List<BufferedImage> dglFrames(String program, Map<String,BufferedImage> inputs, String resultName) {
        return dglWithImages(program, inputs, resultName, Map.of(), tensor -> copyFrames(tensor, true));
    }

    private Latent copyLatent(MemorySegment tensor) {
        long rank = (long) call("sd_tensor_rank", tensor);
        if ((int) call("sd_tensor_type", tensor) != 4 || (rank != 3 && rank != 4) ||
            (long) call("sd_tensor_dim", tensor, 2L) != 4 ||
            (rank == 4 && (long) call("sd_tensor_dim", tensor, 3L) != 1))
            throw new IllegalStateException("Expected one F32 four-channel VAE latent");
        int w = Math.toIntExact((long) call("sd_tensor_dim", tensor, 0L));
        int h = Math.toIntExact((long) call("sd_tensor_dim", tensor, 1L));
        long size = (long) call("sd_tensor_byte_size", tensor);
        if (size != 16L * w * h || size > 67108864) throw new IllegalArgumentException("Invalid VAE output size");
        return new Latent(w, h, ptr("sd_tensor_data", tensor).reinterpret(size).toArray(JAVA_FLOAT));
    }

    public synchronized Latent noise(int width, int height, long seed) {
        ensureOpen();
        String script = "fn main() { g = make_rng(\"cpu\", seed = " + seed + ")\n"
            + "return {value = tensor::randn_like(tensor::zeros([" + width + "," + height + ",4,1]), rng = g)} }";
        try (var arena = Arena.ofConfined()) {
            return dglProgram(arena, script, Map.of(), "value", this::copyLatent);
        }
    }

    public synchronized Latent vaeEncode(String source, BufferedImage image) {
        ensureOpen();
        try (var arena = Arena.ofConfined()) {
            var input = createImageTensor(arena, image, 3);
            try {
                return dglProgram(arena, source, Map.of("input",input), "value", this::copyLatent);
            } finally { call("sd_tensor_free", input); }
        }
    }

    public synchronized BufferedImage vaeDecode(String source, Latent latent) {
        ensureOpen();
        try (var arena = Arena.ofConfined()) {
            var shape = arena.allocateFrom(JAVA_LONG, latent.width, latent.height, 4, 1);
            var input = ptr("sd_tensor_create", 4, 4L, shape, arena.allocateFrom(JAVA_FLOAT, latent.values), 4L * latent.values.length);
            if (input.equals(NULL)) throw new IllegalArgumentException(cString(ptr("sd_get_last_error")));
            try {
                return dglProgram(arena, source, Map.of("input",input), "value", this::copyImage);
            } finally { call("sd_tensor_free", input); }
        }
    }

    private BufferedImage copyImage(MemorySegment tensor) {
        return copyFrames(tensor, false).getFirst();
    }

    private List<BufferedImage> copyFrames(MemorySegment tensor, boolean sequence) {
        if (tensor.equals(NULL) || (int) call("sd_tensor_type", tensor) != 1)
            throw new IllegalStateException("Expected a U8 RGB tensor");
        long rank = (long) call("sd_tensor_rank", tensor);
        if (rank != 3 && !(sequence && rank == 4))
            throw new IllegalStateException("Expected RGB image or frame stack, got rank " + rank);
        int frames = rank == 4 ? Math.toIntExact((long) call("sd_tensor_dim", tensor, 3L)) : 1;
        int channels = Math.toIntExact((long) call("sd_tensor_dim", tensor, 0L));
        int width = Math.toIntExact((long) call("sd_tensor_dim", tensor, 1L));
        int height = Math.toIntExact((long) call("sd_tensor_dim", tensor, 2L));
        long bytes = (long) call("sd_tensor_byte_size", tensor);
        int pixels = Math.multiplyExact(width, height);
        long frameSize = 3L * pixels;
        if (channels != 3 || frames < 1 || bytes != Math.multiplyExact(frameSize, frames) || bytes > 1024L * 1024 * 1024)
            throw new IllegalStateException("Invalid native RGB tensor shape");
        var storage = ptr("sd_tensor_data", tensor).reinterpret(bytes);
        var output = new ArrayList<BufferedImage>(frames);
        for (int frame = 0; frame < frames; ++frame) {
            byte[] rgb = storage.asSlice(frame * frameSize, frameSize).toArray(JAVA_BYTE);
            int[] packed = new int[pixels];
            for (int p = 0, i = 0; p < pixels; p++, i += 3)
                packed[p] = ((rgb[i] & 255) << 16) | ((rgb[i + 1] & 255) << 8) | (rgb[i + 2] & 255);
            var image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            image.setRGB(0, 0, width, height, packed, 0, width);
            output.add(image);
        }
        return List.copyOf(output);
    }

    public void cancel() {
        synchronized (lifetime) {
            if (!standaloneDgl.equals(NULL)) { call("sd_dgl_ctx_cancel", standaloneDgl); return; }
            if (visionRunning) { call("sd_dgl_ctx_cancel", visionCtx); return; }
            if (!ctx.equals(NULL)) {
                if (runningDgl) call("sd_dgl_ctx_cancel", ptr("sd_ctx_get_dgl_ctx", ctx));
                else call("sd_cancel_generation", ctx, 0);
            }
        }
    }

    public synchronized void unload() {
        ensureOpen();
        synchronized (lifetime) {
            if (!ctx.equals(NULL)) call("sd_ctx_free", ctx);
            ctx = NULL;
        }
    }

    @Override public synchronized void close() {
        synchronized (lifetime) {
            if (closed) return;
            unloadClipVision();
            if (!ctx.equals(NULL)) call("sd_ctx_free", ctx);
            ctx = NULL;
            closed = true;
            library.close();
        }
    }
}
