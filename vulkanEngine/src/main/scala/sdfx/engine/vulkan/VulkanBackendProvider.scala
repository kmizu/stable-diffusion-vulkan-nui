package sdfx.engine.vulkan

import java.nio.file.{Files, Path, Paths}
import sdfx.core.config.{AppHome, Settings}
import sdfx.engine.{EngineBackendProvider, EngineLaunch, JvmDryRunWorker, WorkerClasspath}

final class VulkanBackendProvider extends EngineBackendProvider:
  val name = "vulkan"
  def requested(environment: Map[String, String]): Boolean = false
  def usesNativeBackend(settings: Settings, environment: Map[String, String]): Boolean = true

  def launch(home: AppHome, settings: Settings, dryRun: Boolean, environment: Map[String, String]): EngineLaunch =
    val codeSource = Paths.get(getClass.getProtectionDomain.getCodeSource.getLocation.toURI)
    val packaged = Option.when(Files.isRegularFile(codeSource) && codeSource.getParent.endsWith("app")) {
      codeSource.getParent.getParent.resolve("vulkan")
    }
    val runtime = packaged.getOrElse(home.root.toAbsolutePath.resolve("vulkan")).normalize()
    val javaExe = runtime.resolve("jre/bin/java.exe")
    if dryRun then
      val exe = if sys.props("os.name").startsWith("Windows") then "java.exe" else "java"
      val current = Path.of(sys.props("java.home"), "bin", exe)
      val java = Seq(current, javaExe).find(Files.isRegularFile(_)).getOrElse(
        throw IllegalArgumentException("Java runtime missing for Vulkan dry-run"))
      val cp = WorkerClasspath.forResource("sdfx/engine/JvmDryRunWorker.class", JvmDryRunWorker.getClass)
        .getOrElse(throw IllegalArgumentException("JVM dry-run worker missing from classpath"))
      return EngineLaunch(Seq(java.toString, "-Djava.awt.headless=true", "-cp", cp, "sdfx.engine.JvmDryRunWorker"),
        home.root.toAbsolutePath.normalize())
    val jar = runtime.resolve("vulkan-worker.jar")
    val libraries = runtime.resolve("bin")
    require(Files.isRegularFile(javaExe), s"Vulkan Java 25 runtime missing: $javaExe")
    require(Files.isRegularFile(jar), s"Vulkan worker missing: $jar")
    require(Files.isDirectory(libraries), s"Vulkan native library directory missing: $libraries")
    val inheritedPath = environment.collectFirst { case (key, value) if key.equalsIgnoreCase("PATH") => value }
    val workerEnv = Map(
      "SDFX_VULKAN_LIBRARY_DIR" -> libraries.toString,
      "PATH" -> (libraries.toString +: inheritedPath.toList).mkString(java.io.File.pathSeparator)
    ) ++ environment.get("SDFX_VULKAN_DEVICE").orElse(settings.vulkanDevice).map("SDFX_VULKAN_DEVICE" -> _)
    EngineLaunch(
      Seq(javaExe.toString, "--enable-native-access=ALL-UNNAMED", "-jar", jar.toString),
      home.root.toAbsolutePath.normalize(), workerEnv
    )
