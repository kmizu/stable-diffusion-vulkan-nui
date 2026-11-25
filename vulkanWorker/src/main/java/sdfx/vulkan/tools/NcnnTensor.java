package sdfx.vulkan.tools;

import java.io.*;
import java.nio.*;
import java.nio.file.Path;
import java.util.concurrent.*;

/** Run one planar float32 ncnn model on the selected Vulkan adapter. */
public final class NcnnTensor {
    public static float[] run(Path bridge, String device, Path model, String part, float[] data, int width, int height, int outputs, Runnable check) throws Exception {
        check.run();
        int channels = data.length / (width * height);
        Process proc = new ProcessBuilder(bridge.toString(), device,
                model.resolve(part + ".ncnn.param").toString(), model.resolve(part + ".ncnn.bin").toString(),
                "in0", "out0", Integer.toString(width), Integer.toString(height), Integer.toString(channels))
                .redirectError(ProcessBuilder.Redirect.INHERIT).start();
        try {
            CompletableFuture<float[]> io = CompletableFuture.supplyAsync(() -> {
                byte[] bytes = new byte[65536];
                FloatBuffer floats = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
                try {
                    try (OutputStream out = proc.getOutputStream()) {
                        for (int pos = 0; pos < data.length; pos += floats.capacity()) {
                            int count = Math.min(floats.capacity(), data.length-pos);
                            floats.clear().put(data,pos,count);
                            out.write(bytes,0,count*4);
                        }
                    }
                    float[] result = new float[outputs];
                    try (DataInputStream in = new DataInputStream(proc.getInputStream())) {
                        for (int pos = 0; pos < outputs; pos += floats.capacity()) {
                            int count = Math.min(floats.capacity(), outputs-pos);
                            in.readFully(bytes,0,count*4);
                            floats.clear().get(result,pos,count);
                        }
                        if (in.read() != -1) throw new IOException("Unexpected ncnn output shape");
                    }
                    return result;
                } catch (IOException err) { throw new UncheckedIOException(err); }
            });
            while (!proc.waitFor(100, TimeUnit.MILLISECONDS)) check.run();
            if (proc.exitValue() != 0) throw new IOException("ncnn " + part + " exited with status " + proc.exitValue());
            return io.get(10,TimeUnit.SECONDS);
        } finally { proc.destroyForcibly(); }
    }
}
