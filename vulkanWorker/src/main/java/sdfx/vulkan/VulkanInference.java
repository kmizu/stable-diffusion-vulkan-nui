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
    @FunctionalInterface public interface Progress {
        void update(String stage, long completed, long total);
    }

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
    private boolean closed;

    public VulkanInference(Path nativeDir) {
        Path dir = nativeDir.toAbsolutePath().normalize();
        try {
            for (String name : List.of("ggml-base", "ggml", "libsharpyuv", "libwebp", "libwebpdemux", "libwebpmux", "webm"))
                SymbolLookup.libraryLookup(dir.resolve(name + ".dll"), library);
            var ggml = SymbolLookup.libraryLookup(dir.resolve("ggml.dll"), library);
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
            bind("sd_opt_set_bool", JAVA_INT, ADDRESS, ADDRESS, JAVA_BOOLEAN);
            bind("sd_opt_set_int", JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG);
            bind("sd_opt_set_float", JAVA_INT, ADDRESS, ADDRESS, JAVA_DOUBLE);
            bind("sd_ctx_create", JAVA_INT, ADDRESS, ADDRESS);
            bind("sd_ctx_supports_image_generation", JAVA_BOOLEAN, ADDRESS);
            bind("sd_get_model_version_name", ADDRESS, ADDRESS);
            bind("sd_ctx_free", null, ADDRESS);
            bind("sd_generate_image", JAVA_INT, ADDRESS, ADDRESS, ADDRESS);
            bind("sd_set_generation_progress_callback", null, ADDRESS, ADDRESS, ADDRESS);
            bind("sd_cancel_generation", null, ADDRESS, JAVA_INT);
            bind("sd_result_free", null, ADDRESS);
            bind("sd_result_get_image", ADDRESS, ADDRESS, JAVA_LONG);
            bind("sd_tensor_dim", JAVA_LONG, ADDRESS, JAVA_LONG);
            bind("sd_tensor_byte_size", JAVA_LONG, ADDRESS);
            bind("sd_tensor_type", JAVA_INT, ADDRESS);
            bind("sd_tensor_rank", JAVA_LONG, ADDRESS);
            bind("sd_tensor_data", ADDRESS, ADDRESS);
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
        ensureOpen();
        if (!Files.isRegularFile(model)) throw new IllegalArgumentException("Missing model file: " + model);
        if (devices().stream().noneMatch(d -> d.id().equals(deviceId)))
            throw new IllegalArgumentException("Unknown Vulkan device: " + deviceId);
        try (var a = Arena.ofConfined()) {
            var opt = newOptions("sd_ctx_opt_create");
            try {
                string(a, opt, "model_path", model.toAbsolutePath().normalize().toString());
                string(a, opt, "backend", deviceId);
                if (!weightType.isBlank()) string(a, opt, "wtype", weightType);
                string(a, opt, "params_backend", deviceId);
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

    public synchronized String modelVersion() {
        ensureOpen();
        if (ctx.equals(NULL)) throw new IllegalStateException("No model loaded");
        return cString(ptr("sd_get_model_version_name", ctx));
    }

    public BufferedImage generate(String prompt, String negativePrompt, int width, int height, int steps,
            double cfg, long seed, String sampler, String scheduler, Progress progress) {
        return generate(prompt, negativePrompt, width, height, steps, cfg, seed, sampler, scheduler, 1, progress);
    }

    public synchronized BufferedImage generate(String prompt, String negativePrompt, int width, int height, int steps,
            double cfg, long seed, String sampler, String scheduler, int clipSkip, Progress progress) {
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
                integer(a, opt, "clip_skip", clipSkip);
                integer(a, opt, "seed", seed);
                check(call("sd_opt_set_float", opt, a.allocateFrom("sample.guidance.txt_cfg"), cfg));
                string(a, opt, "sample.method", sampler);
                string(a, opt, "sample.scheduler", scheduler);
                var result = a.allocate(ADDRESS);
                var callback = progress == null ? NULL : LINKER.upcallStub(EVENT_HANDLER.bindTo(progress), PROGRESS, a);
                call("sd_set_generation_progress_callback", ctx, callback, NULL);
                try {
                    check(call("sd_generate_image", ctx, opt, result));
                    var handle = result.get(ADDRESS, 0);
                    try {
                        return copyImage(ptr("sd_result_get_image", handle, 0L));
                    } finally { call("sd_result_free", handle); }
                } finally { call("sd_set_generation_progress_callback", ctx, NULL, NULL); }
            } finally { call("sd_opt_free", opt); }
        }
    }

    private BufferedImage copyImage(MemorySegment tensor) {
        if (tensor.equals(NULL) || (int) call("sd_tensor_type", tensor) != 1 || (long) call("sd_tensor_rank", tensor) != 3)
            throw new IllegalStateException("Expected a U8 image tensor");
        int channels = Math.toIntExact((long) call("sd_tensor_dim", tensor, 0L));
        int width = Math.toIntExact((long) call("sd_tensor_dim", tensor, 1L));
        int height = Math.toIntExact((long) call("sd_tensor_dim", tensor, 2L));
        long bytes = (long) call("sd_tensor_byte_size", tensor);
        int pixels = Math.multiplyExact(width, height);
        if (channels != 3 || bytes != 3L * pixels) throw new IllegalStateException("Invalid native RGB tensor shape");
        byte[] rgb = ptr("sd_tensor_data", tensor).reinterpret(bytes).toArray(JAVA_BYTE);
        int[] packed = new int[pixels];
        for (int p = 0, i = 0; p < pixels; p++, i += 3)
            packed[p] = ((rgb[i] & 255) << 16) | ((rgb[i + 1] & 255) << 8) | (rgb[i + 2] & 255);
        var image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        image.setRGB(0, 0, width, height, packed, 0, width);
        return image;
    }

    public void cancel() {
        synchronized (lifetime) {
            if (!ctx.equals(NULL)) call("sd_cancel_generation", ctx, 0);
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
            if (!ctx.equals(NULL)) call("sd_ctx_free", ctx);
            ctx = NULL;
            closed = true;
            library.close();
        }
    }
}
