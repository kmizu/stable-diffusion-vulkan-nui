package sdfx.vulkan;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.zip.ZipFile;

/** A1111 VAE approximation. Decode the official fixed state dictionary's raw float storages
 * directly, without running a pickle interpreter. No native runtime is needed for the small CNN. */
public final class ApproxVaeDecoder {
    private static final int[] CHANNELS = {4, 8, 16, 32, 64, 32, 16, 8, 3};
    private static final int[] KERNELS = {7, 5, 3, 3, 3, 3, 3, 3};
    private static final String STRUCTURE = "467e0c221779a7fbc3f41687e92ab02f27fc55c176074280fdf634c17b157e32";
    private final float[][] weights = new float[8][];
    private final float[][] biases = new float[8][];

    public ApproxVaeDecoder(Path path) throws IOException, NoSuchAlgorithmException {
        if (!Files.isRegularFile(path) || Files.size(path) > 16L * 1024 * 1024)
            throw new IllegalArgumentException("Missing or oversized Approx NN weights");
        try (var zip = new ZipFile(path.toFile())) {
            byte[] description = read(zip, "model/data.pkl", 1693);
            if (!HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(description)).equals(STRUCTURE))
                throw new IllegalArgumentException("Expected an A1111 Approx NN state dictionary");
            for (int i = 0; i < 8; i++) {
                int size = CHANNELS[i] * CHANNELS[i+1] * KERNELS[i] * KERNELS[i];
                weights[i] = floats(read(zip, "model/data/" + 2*i, 4*size));
                biases[i] = floats(read(zip, "model/data/" + (2*i+1), 4*CHANNELS[i+1]));
            }
        }
    }

    private static byte[] read(ZipFile zip, String name, int size) throws IOException {
        var entry = zip.getEntry(name);
        if (entry == null || entry.getSize() != size) throw new IllegalArgumentException("Invalid Approx NN storage: " + name);
        try (var in = zip.getInputStream(entry)) {
            byte[] bytes = in.readNBytes(size + 1);
            if (bytes.length != size) throw new IllegalArgumentException("Truncated Approx NN storage");
            return bytes;
        }
    }

    private static float[] floats(byte[] bytes) {
        float[] result = new float[bytes.length/4];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(result);
        for (float f : result) if (!Float.isFinite(f)) throw new IllegalArgumentException("Nonfinite Approx NN parameter");
        return result;
    }

    public BufferedImage decode(VulkanInference.Latent latent, Runnable check) {
        int width = 2 * latent.width() + 22, height = 2 * latent.height() + 22;
        if ((long)width * height * 64 > 134217728L) throw new IllegalArgumentException("Approx NN input exceeds limit");
        float[] input = new float[width*height*4];
        for (int c=0; c<4; c++) for (int y=11; y<height-11; y++) {
            check.run();
            for (int x=11; x<width-11; x++)
                input[(c*height+y)*width+x] = latent.values()[(c*latent.height()+(y-11)/2)*latent.width()+(x-11)/2];
        }
        for (int layer=0; layer<8; layer++) {
            int kernel = KERNELS[layer], ow = width-kernel+1, oh = height-kernel+1;
            float[] output = new float[CHANNELS[layer+1]*oh*ow];
            for (int dst=0; dst<CHANNELS[layer+1]; dst++) {
                int base = dst*ow*oh;
                java.util.Arrays.fill(output,base,base+ow*oh,biases[layer][dst]);
                for (int src=0; src<CHANNELS[layer]; src++) {
                    check.run();
                    for (int ky=0; ky<kernel; ky++) for (int kx=0; kx<kernel; kx++) {
                        float weight = weights[layer][((dst*CHANNELS[layer]+src)*kernel+ky)*kernel+kx];
                        for (int y=0; y<oh; y++) {
                            int row = (src*height+y+ky)*width+kx;
                            int destRow = base+y*ow;
                            for (int x=0; x<ow; x++) output[destRow+x] += input[row+x]*weight;
                        }
                    }
                }
                for (int index=base; index<base+ow*oh; index++) if (output[index]<0) output[index]*=.1f;
            }
            width=ow; height=oh; input=output;
        }
        var image = new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB);
        int plane=width*height;
        int[] pixels = new int[plane];
        for (int i=0; i<plane; i++) for (int c=0; c<3; c++)
            pixels[i] = (pixels[i]<<8) | (int)((Math.clamp(input[c*plane+i],-1f,1f)+1f)*127.5f);
        image.setRGB(0,0,width,height,pixels,0,width);
        return image;
    }
}
