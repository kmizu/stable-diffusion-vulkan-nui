// Adapted from kmizu/scala-torch (BSD-3-Clause). See META-INF/LICENSE-scala-torch.txt.
package sdfx.vulkan.tools

import java.nio.file.{Files,Path}
import scala.util.Using
import CheckpointIO.*
import NativeJson.*

object NativeModelConvert:
  private val unet="model.diffusion_model."
  private val junk=Vector("optimizer","model_ema.","lr_scheduler","global_step","epoch","loss")
  def execute(p: Map[String,NativeJson],ctx: NativeWorkContext): Map[String,NativeJson] =
    require((p.keySet--Set("model","output","precision","ema","fix_clip","drop_junk","bake_vae","overwrite","output_dir")).isEmpty,"unknown conversion parameters")
    def str(k: String,default: String=""): String=p.get(k) match
      case None | Some(Null) => default
      case Some(Str(s)) => s
      case _ => throw IllegalArgumentException(s"$k must be a string")
    def bool(k: String,default: Boolean): Boolean=p.get(k) match
      case None | Some(Null) => default
      case Some(Bool(b)) => b
      case _ => throw IllegalArgumentException(s"$k must be boolean")
    def file(k: String): Path =
      val name=str(k)
      require(name.nonEmpty && name.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") && Files.isRegularFile(Path.of(name)),s"$k must be a safetensors file")
      Path.of(name).toRealPath()
    val source=file("model")
    val vaeName=str("bake_vae")
    val vaePath=Option.when(vaeName.nonEmpty) {
      val path=Path.of(vaeName)
      require(Files.isRegularFile(path),"bake_vae must be a file")
      path.toRealPath()
    }
    val output=Path.of(str("output")).toAbsolutePath.normalize()
    require(output.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") && output.getParent!=null && Files.isDirectory(output.getParent),"invalid output path")
    val realTarget=output.getParent.toRealPath().resolve(output.getFileName).normalize()
    require((Vector(source)++vaePath.toVector).forall(p=>p!=realTarget && (!Files.exists(output) || !Files.isSameFile(p,output))),"output would overwrite an input")
    val overwrite=bool("overwrite",false)
    require(overwrite || !Files.exists(output),"output already exists")
    val precision=str("precision","fp16").toLowerCase(java.util.Locale.ROOT)
    val target=precision match
      case "keep" => None
      case "fp16" => Some("F16")
      case "bf16" => Some("BF16")
      case "fp32" => Some("F32")
      case _ => throw IllegalArgumentException("invalid precision")
    val ema=str("ema","no-ema");require(Set("no-ema","ema-only","keep")(ema),"invalid EMA mode")
    val fix=bool("fix_clip",true);val drop=bool("drop_junk",true)
    Using.resource(CheckpointIO.open(source)) { src =>
      val vae=vaePath.filter(_.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors")).map(CheckpointIO.open(_))
      require(vaePath.isEmpty || vae.nonEmpty, "VAE conversion requires a safetensors VAE")
      try
        val vaeKeys=vae.map(_.tensors.keysIterator.filterNot(k=>k.startsWith("loss") || k.startsWith("model_ema"))
          .map(k=>(k.stripPrefix("first_stage_model."),k)).toMap).getOrElse(Map.empty[String,String])
        val keys=src.tensors.keys.toVector.sorted
        var dropped=0;var used=0;var fixed=0
        val plans=keys.flatMap { key =>
          if key.startsWith("model_ema.") && ema!="keep" || drop && !key.startsWith("model_ema.") && junk.exists(key.startsWith) then
            dropped+=1;None
          else
            val emaKey=if ema=="ema-only" && key.startsWith(unet) then
              "model_ema."+("diffusion_model"+key.drop(unet.length)).replace(".","") else ""
            val selected=if emaKey.nonEmpty && src.tensors.contains(emaKey) then {used+=1;emaKey} else key
            val sourceInfo=src.tensors(selected)
            val position=fix && (key=="position_ids" || key.endsWith(".position_ids")) && !key.startsWith("model_ema.")
            if position then fixed+=1
            val vaeKey=if !position && key.startsWith("first_stage_model.") then
              vaeKeys.get(key.stripPrefix("first_stage_model.")).filter(k=>vae.get.tensors(k).shape==sourceInfo.shape) else None
            val selectedFile=if vaeKey.nonEmpty then vae.get else src
            val selectedKey=vaeKey.getOrElse(selected)
            val info=selectedFile.tensors(selectedKey)
            val plan = if position then positions(info)
              else cast(selectedFile, info, target.filter(_ => floatTypes(info.dtype)).getOrElse(info.dtype))
            Some(key -> plan)
        }.toVector
        require(ema!="ema-only" || used>0,"ema-only: this checkpoint has no EMA weights")
        val recipe=Obj(Map("source"->Str(source.getFileName.toString),"precision"->Str(precision),"ema"->Str(ema),
          "fix_clip"->Bool(fix),"drop_junk"->Bool(drop),"bake_vae"->vaePath.map(p=>Str(p.getFileName.toString)).getOrElse(Null)))
        write(output,plans,Map("format"->"pt","sdfx_convert"->NativeJson.render(recipe)),overwrite,ctx,"convert")
        ctx.event("progress",Map("step"->Number(keys.size),"total"->Number(keys.size),"pass"->Str("convert")))
        Map("path"->Str(output.toString),"keys"->Number(plans.size),"dropped"->Number(dropped),
          "ema_weights_used"->Number(used),"clip_fixed"->Number(fixed),"vae_baked"->Bool(vaeKeys.nonEmpty))
      finally vae.foreach(_.close())
    }
