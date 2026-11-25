package sdfx.vulkan.tools

import java.nio.file.{Files,Path}
import java.nio.file.attribute.BasicFileAttributes
import scala.util.Using

/** f296cb0 reads CompVis SD1 ControlNets and also Diffusers SDXL ControlNets. Normalize
  * SD1 Diffusers names once per version of the source, preserving dtype, shape and weight bytes.
  */
private[vulkan] final class VulkanControlModels extends AutoCloseable:
  private case class Cached(key: AnyRef, size: Long, time: java.nio.file.attribute.FileTime, target: Path)
  private val cache = scala.collection.mutable.Map.empty[Path,Cached]
  private val temporary = scala.collection.mutable.ArrayBuffer.empty[Path]
  private val down = """down_blocks\.(\d+)\.(resnets|attentions)\.(\d+)\.(.+)""".r
  private val sampler = """down_blocks\.(\d+)\.downsamplers\.0\.conv\.(.+)""".r
  private val cond = """controlnet_cond_embedding\.blocks\.(\d+)\.(.+)""".r
  private val zero = """controlnet_down_blocks\.(\d+)\.(.+)""".r
  private val resnet = Vector("norm1." -> "in_layers.0.", "conv1." -> "in_layers.2.",
    "time_emb_proj." -> "emb_layers.1.", "norm2." -> "out_layers.0.", "conv2." -> "out_layers.3.",
    "conv_shortcut." -> "skip_connection.")
  private def prefix(name: String, rules: Seq[(String,String)]): String =
    rules.collectFirst { case (from,to) if name.startsWith(from) => to + name.drop(from.length) }.getOrElse(name)
  private def convertKey(key: String): String = key match
    case down(b,"resnets",r,k) => s"input_blocks.${1+3*b.toInt+r.toInt}.0." + prefix(k,resnet)
    case down(b,"attentions",r,k) => s"input_blocks.${1+3*b.toInt+r.toInt}.1.$k"
    case sampler(b,k) => s"input_blocks.${3*(b.toInt+1)}.0.op.$k"
    case cond(i,k) => s"input_hint_block.${2*(i.toInt+1)}.$k"
    case zero(i,k) => s"zero_convs.$i.0.$k"
    case k if k.startsWith("mid_block.resnets.0.") => "middle_block.0." + prefix(k.stripPrefix("mid_block.resnets.0."),resnet)
    case k if k.startsWith("mid_block.resnets.1.") => "middle_block.2." + prefix(k.stripPrefix("mid_block.resnets.1."),resnet)
    case k => prefix(k, Vector("conv_in." -> "input_blocks.0.0.", "time_embedding.linear_1." -> "time_embed.0.",
      "time_embedding.linear_2." -> "time_embed.2.", "mid_block.attentions.0." -> "middle_block.1.",
      "controlnet_cond_embedding.conv_in." -> "input_hint_block.0.", "controlnet_cond_embedding.conv_out." -> "input_hint_block.14.",
      "controlnet_mid_block." -> "middle_block_out.0."))

  def resolve(selected: String, family: String, cancelled: () => Boolean): String =
    val entry = Path.of(selected).toAbsolutePath.normalize()
    val path = if !Files.isDirectory(entry) then entry else
      Seq("diffusion_pytorch_model.fp16.safetensors","diffusion_pytorch_model.safetensors")
        .map(entry.resolve).find(Files.isRegularFile(_)).getOrElse(throw IllegalArgumentException(s"No ControlNet safetensors in $entry"))
    require(Files.isRegularFile(path),s"Missing ControlNet: $path")
    if family != "sd15" || !path.getFileName.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") then return path.toString
    val stamp = Files.readAttributes(path,classOf[BasicFileAttributes])
    cache.get(path).filter(c => c.key == stamp.fileKey && c.size == stamp.size && c.time == stamp.lastModifiedTime).fold {
      val result = Using.resource(CheckpointIO.open(path)) { source =>
        if !source.tensors.contains("controlnet_down_blocks.11.weight") then path else
          val dir = Files.createTempDirectory("sdv-controlnet-")
          val output = dir.resolve(path.getFileName)
          try
            val plans = source.tensors.toVector.map((k,t) => ("control_model." + convertKey(k)) -> CheckpointIO.copy(source,t))
            require(plans.map(_._1).distinct.size == plans.size,"Ambiguous ControlNet tensor names")
            val work = new NativeWorkContext(cancelled,() => (),(_,_) => ())
            CheckpointIO.write(output,plans,Map.empty,false,work,"controlnet")
            temporary ++= Seq(output,dir)
            output
          catch case e: Throwable =>
            Files.deleteIfExists(output)
            Files.deleteIfExists(dir)
            throw e
      }
      cache(path) = Cached(stamp.fileKey,stamp.size,stamp.lastModifiedTime,result)
      result.toString
    }(_.target.toString)
  def close(): Unit = temporary.foreach(Files.deleteIfExists)
