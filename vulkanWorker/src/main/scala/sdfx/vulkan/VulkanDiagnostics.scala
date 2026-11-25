package sdfx.vulkan

import java.lang.management.ManagementFactory
import java.nio.file.{Files, Path}
import java.time.Instant
import scala.util.Using
import sdfx.core.imagebrowser.BrowserFiles
import sdfx.core.protocol.*
import ujson.{Arr, Obj, Value}

private[vulkan] object VulkanDiagnostics:
  private val mib = 1048576.0
  private val started = ManagementFactory.getRuntimeMXBean.getStartTime

  def memory(sample: VulkanInference.DeviceMemory): MemoryInfo =
    val used = Option.when(sample.total > 0 && sample.free >= 0 && sample.free <= sample.total)(
      (sample.total - sample.free) / mib)
    MemoryInfo(totalMb = sample.total / mib, deviceUsedMb = used)

  def status(device: VulkanInference.Device, sample: VulkanInference.DeviceMemory, model: Option[LoadedModel]): Obj =
    val info = memory(sample)
    val heap = ManagementFactory.getMemoryMXBean.getHeapMemoryUsage
    val gpu = Obj("device" -> device.name(), "id" -> device.id(), "total_mb" -> info.totalMb,
      "free_mb" -> sample.free / mib, "scope" -> "whole device")
    info.deviceUsedMb.foreach(v => gpu("used_mb") = v)
    val mem = Obj("gpu" -> gpu, "jvm_heap_used_mb" -> heap.getUsed / mib,
      "jvm_heap_committed_mb" -> heap.getCommitted / mib)
    ManagementFactory.getOperatingSystemMXBean match
      case os: com.sun.management.OperatingSystemMXBean =>
        mem("ram_total_mb") = os.getTotalMemorySize / mib
        mem("ram_available_mb") = os.getFreeMemorySize / mib
      case _ => ()
    val now = Instant.now()
    Obj("timestamp" -> now.toString, "started_at" -> Instant.ofEpochMilli(started).toString,
      "uptime_seconds" -> (now.toEpochMilli - started) / 1000.0, "loaded" -> model.nonEmpty,
      "model" -> model.map(Json.toJs(_)).getOrElse(Obj()), "memory" -> mem, "gpu" -> gpu,
      "device" -> VulkanProtocol.display(device), "dry_run" -> false,
      "runtime" -> Obj("backend" -> "vulkan", "pipeline" -> "stable-diffusion.cpp"),
      "version" -> Obj("app" -> "SDV-NUI", "engine" -> "stable-diffusion.cpp", "hash" -> "f296cb0",
        "url" -> "https://github.com/kmizu/stable-diffusion-vulkan-nui"),
      "platform" -> Obj("arch" -> sys.props("os.arch"), "system" -> sys.props("os.name"),
        "release" -> sys.props("os.version"), "cpu" -> sys.env.getOrElse("PROCESSOR_IDENTIFIER", sys.props("os.arch"))),
      "libraries" -> Obj("java" -> sys.props("java.version")))

  def metadata(status: Value, fields: Seq[String]): Obj =
    val aliases = Map("backend" -> "runtime.backend", "pipeline" -> "runtime.pipeline")
    val roots = Set("version", "platform", "gpu", "runtime", "libraries")
    val result = Obj()
    fields.distinct.foreach { field =>
      aliases.getOrElse(field, field).split("\\.").toList match
        case group :: key :: Nil if roots(group) =>
          status.obj.get(group).flatMap(_.obj.get(key)).filter(_.strOpt.isDefined).foreach(result(field) = _)
        case _ => ()
    }
    result

  def inventory(params: Value, status: Obj, features: Value, check: () => Unit): Obj =
    val hashes = params.obj.get("hashes").exists(_.bool)
    val dirs = params.obj.getOrElse("dirs", Obj())
    val networks = Obj()
    for kind <- Seq("checkpoints", "lora", "vae", "embeddings", "hypernetworks") do
      val sources = dirs.obj.get(kind).toSeq.flatMap(_.arr).map(v => Path.of(v.str).toAbsolutePath.normalize())
      val seen = scala.collection.mutable.Set.empty[Path]
      val rows = sources.filter(Files.isDirectory(_)).flatMap { root =>
        Using.resource(Files.walk(root)) { walk =>
          import scala.jdk.CollectionConverters.*
          walk.iterator().asScala.filter { path =>
            check()
            val name = path.getFileName.toString.toLowerCase(java.util.Locale.ROOT)
            Files.isRegularFile(path) && (if kind == "checkpoints" then VulkanProtocol.checkpoint(path)
              else Seq(".safetensors", ".ckpt", ".pt", ".pth", ".bin").exists(name.endsWith)) && seen.add(path.toRealPath())
          }.map { path =>
            val row = Obj("name" -> root.relativize(path).toString, "kind" -> kind, "size_mb" -> Files.size(path) / mib)
            if hashes then row("hash") = BrowserFiles.sha256(path).take(10)
            row
          }.toVector
        }
      }
      networks(kind) = Arr.from(rows.sortBy(_("name").str))
    status("networks") = networks
    status("features") = features
    status("schedulers") = Json.toJs(VulkanProtocol.samplerList)
    status
