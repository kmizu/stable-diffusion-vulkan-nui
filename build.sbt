ThisBuild / scalaVersion := "3.3.6"
ThisBuild / version := "0.1"
ThisBuild / organization := "sdfx"
ThisBuild / scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-encoding", "utf8")

val upickleV = "4.1.0"
val munitV = "1.1.1"
val fxV = "21.0.6"

lazy val fxOs: String = {
  val n = System.getProperty("os.name").toLowerCase
  val arch = System.getProperty("os.arch")
  if (n.contains("win")) "win"
  else if (n.contains("mac")) { if (arch == "aarch64") "mac-aarch64" else "mac" }
  else "linux"
}
lazy val fxDeps = Seq("base", "graphics", "controls", "swing", "media").map(m => "org.openjfx" % s"javafx-$m" % fxV classifier fxOs)

lazy val commonSettings = Seq(
  libraryDependencies ++= Seq(
    "com.lihaoyi" %% "upickle" % upickleV,
    "com.twelvemonkeys.imageio" % "imageio-webp" % "3.12.0",
    "org.scalameta" %% "munit" % munitV % Test
  ),
  // sbt 2 puts jars on the classpath by default; tests read fixture resources as files.
  exportJars := false,
  Test / fork := true,
  Test / baseDirectory := (ThisBuild / baseDirectory).value,
  Test / parallelExecution := false,
  Compile / run / fork := true,
  Compile / run / baseDirectory := (ThisBuild / baseDirectory).value,
  run / connectInput := true
)

lazy val core = project.settings(commonSettings, name := "sdfx-core", Test / javaOptions ++= Seq("-Djava.awt.headless=true", "-Dfile.encoding=UTF-8"))

lazy val engine = project.dependsOn(core).settings(
  commonSettings,
  name := "sdfx-engine",
  libraryDependencies += "com.microsoft.onnxruntime" % "onnxruntime" % "1.20.0"
)

lazy val vulkanEngine = project.dependsOn(engine).settings(commonSettings, name := "sdfx-vulkan-engine")

// The Java 25 FFM classes live in their own process and never enter the desktop jar.
lazy val vulkanWorker = project.dependsOn(core).settings(
  commonSettings,
  name := "sdfx-vulkan-worker",
  libraryDependencies += "com.fasterxml.jackson.core" % "jackson-core" % "2.20.1",
  libraryDependencies += "org.apache.commons" % "commons-math3" % "3.6.1",
  Compile / resourceGenerators += Def.task {
    val labels = (ThisBuild / baseDirectory).value / "python" / "sdfx" / "data" / "clip_interrogator"
    val out = (Compile / resourceManaged).value / "clip-interrogator"
    val notice = out / "LICENSE.txt"
    IO.copyFile((ThisBuild / baseDirectory).value / "licenses" / "pharmapsychotic__clip-interrogator.txt", notice)
    Seq("artists", "mediums", "movements", "flavors", "negative").map { kind =>
      val name = kind + ".txt"
      IO.copyFile(labels / name, out / name)
      out / name
    } :+ notice
  }.taskValue,
  Compile / javacOptions ++= Seq("--release", "25"),
  Compile / mainClass := Some("sdfx.vulkan.VulkanWorker"),
  assembly / mainClass := Some("sdfx.vulkan.VulkanWorker"),
  assembly / assemblyOutputPath := (ThisBuild / baseDirectory).value / "target" / "vulkan-worker.jar"
)

lazy val plugins = project.dependsOn(core, engine).settings(commonSettings, name := "sdfx-plugins",
  libraryDependencies ++= Seq("com.google.zxing" % "core" % "3.5.3", "com.google.zxing" % "javase" % "3.5.3"))

lazy val mcp = project
  .dependsOn(core, engine, plugins, vulkanEngine % "runtime->compile")
  .settings(commonSettings, name := "sdfx-mcp")

/** Dev-only JVM client built on the official MCP Java SDK; replaces the Python scripts for acceptance runs. */
lazy val mcpClient = project
  .in(file("tools/mcp-client"))
  .settings(
    name := "sdfx-mcp-client",
    scalaVersion := (core / scalaVersion).value,
    libraryDependencies ++= Seq(
      "io.modelcontextprotocol.sdk" % "mcp" % "1.1.5",
      "com.lihaoyi" %% "upickle" % upickleV
    ),
    exportJars := false
  )

lazy val ui = project
  .dependsOn(core, engine, plugins, mcp, vulkanEngine % "runtime->compile")
  .settings(
    commonSettings,
    name := "sdfx-ui",
    libraryDependencies ++= fxDeps,
    Compile / mainClass := Some("sdfx.ui.Launcher"),
    run / javaOptions ++= Seq("-Dprism.allowhidpi=true", "-Xmx2g", "-Dfile.encoding=UTF-8"),
    Test / javaOptions ++= Seq("-Djava.awt.headless=true"),
    assembly / mainClass := Some("sdfx.ui.Launcher"),
    assembly / assemblyJarName := "stable-diffusion-nui.jar",
    assembly / assemblyOutputPath := (ThisBuild / baseDirectory).value / "target" / "stable-diffusion-nui.jar",
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", "services", _*) => MergeStrategy.concat
      case PathList("META-INF", _*)              => MergeStrategy.discard
      case x if x.endsWith("module-info.class")  => MergeStrategy.discard
      case _                                     => MergeStrategy.first
    }
  )

lazy val root = (project in file("."))
  .aggregate(core, engine, plugins, mcp, ui, vulkanEngine, vulkanWorker)
  .settings(name := "stable-diffusion-vulkan-nui", publish / skip := true)
