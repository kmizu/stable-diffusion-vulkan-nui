import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.graalvm.polyglot.Context;

/** Separate JVM for the explicitly enabled Custom Code plugin. */
public final class VulkanScriptHost {
    public static void main(String[] args) throws Exception {
        if (!"1".equals(System.getenv("SDFX_ALLOW_CODE"))) {
            throw new IllegalStateException("Custom code is disabled");
        }
        try (var ctx = Context.newBuilder("python").allowAllAccess(true)
                .option("engine.WarnInterpreterOnly", "false").build()) {
            var script = Files.readString(Path.of(args[0]), StandardCharsets.UTF_8);
            ctx.eval("python", script);
        }
    }
}
