package sdfx.vulkan;

import java.nio.file.Path;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CountDownLatch;
import javax.imageio.ImageIO;

/** Standalone real-driver smoke. Args: nativeDir model VulkanN png [size steps [cancel-first]]. */
public final class VulkanSmoke {
    public static void main(String[] args) throws Exception {
        try (var engine = new VulkanInference(Path.of(args[0]))) {
            System.err.println(engine.devices());
            engine.load(Path.of(args[1]), args[2]);
            System.err.println("Model: " + engine.modelVersion());
            int size = args.length > 4 ? Integer.parseInt(args[4]) : 512;
            int steps = args.length > 5 ? Integer.parseInt(args[5]) : 8;
            if (args.length > 6 && args[6].equals("cancel-first")) verifyCancel(engine);
            long start = System.nanoTime();
            var image = engine.generate("a watercolor painting of a red fox sitting in a spring meadow, green grass, blue sky",
                "low quality, blurry", size, size, steps, 7, 12345L, "euler", "karras",
                (stage, current, total) -> System.err.printf("%s %d/%d%n", stage, current, total));
            System.err.printf("Generation completed in %.2fs%n", (System.nanoTime() - start) / 1e9);
            if (!ImageIO.write(image, "png", Path.of(args[3]).toFile())) throw new IllegalStateException("No PNG writer");
        }
    }

    private static void verifyCancel(VulkanInference engine) throws InterruptedException {
        var sampling = new CountDownLatch(1);
        Thread requester = Thread.ofPlatform().start(() -> {
            try {
                sampling.await();
                engine.cancel();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        try {
            engine.generate("an orange fox", "", 256, 256, 12, 7, 12345L, "euler", "karras",
                (stage, step, total) -> {
                    if (stage.equals("sampling") && step == 1) sampling.countDown();
                });
            throw new AssertionError("Native generation completed despite cancellation");
        } catch (CancellationException expected) {
            if (sampling.getCount() != 0) throw new AssertionError("Cancelled without request", expected);
            System.err.println("Concurrent cancellation acknowledged. Reusing loaded context.");
        } finally {
            requester.interrupt();
            requester.join();
        }
    }
}
