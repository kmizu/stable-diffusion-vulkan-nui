package sdfx.vulkan.tools

import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{FileAlreadyExistsException,Files,Path}
import java.util.zip.ZipFile
import scala.util.Using
import sdfx.core.imagebrowser.BrowserFiles

private[vulkan] object VulkanBaseClip:
  private val versions=Map(
    16->("57c216476eefef5ab752ec549e440a49ae4ae5f3","ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0","b840492714fee524d69b5b2f8206b2398cc5d5034b55d04c1aa05218bc2902a5"),
    32->("3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268","a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f","fe12b9aac6c80b4c87637656e652e15b3f879c96e45e13ba267ec086a61b2174"))
  // The two manifests list shapes, strides and storage slices from the SHA-pinned OpenAI releases.
  // Reading those slices avoids executing or interpreting third-party pickle programs.
  def program(root:Path,patch:Int,device:String,check:()=>Unit):String =
    val (revision,sha,convertedSha)=versions(patch)
    val name=s"clip-vit-base-patch$patch"
    val dir=Files.createDirectories(root.resolve("openai__"+name).resolve(revision))
    val out=dir.resolve("vision-v1.safetensors")
    if !Files.isRegularFile(out) then
      val url=s"https://huggingface.co/openai/$name/resolve/$revision/pytorch_model.bin"
      val archive=VulkanClip.asset(dir,"pytorch_model.bin",url,sha,check,(_,_,_)=>())
      Using.resource(new ZipFile(archive.toFile)) { zip =>
        val rows=Using.resource(scala.io.Source.fromResource(s"sdfx/clip/openai-clip-b$patch.txt"))(_.getLines().toVector)
        val plans=rows.map { line =>
          val parts=line.split(' ')
          val shape=parts(3).split(',').map(_.toLong).toVector
          val size=shape.product.toInt
          val read=() => Using.resource(zip.getInputStream(zip.getEntry("archive/data/"+parts(1)))) { in =>
            in.skipNBytes(parts(2).toLong)
            val raw=in.readNBytes(size*4)
            require(raw.length==size*4)
            if shape.length==2 && parts(4).startsWith("1,") && shape(1)>1 then
              val source=ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer()
              val packed=ByteBuffer.allocate(raw.length).order(ByteOrder.LITTLE_ENDIAN)
              for r<-0 until shape(0).toInt; c<-0 until shape(1).toInt do packed.putInt(source.get(c*shape(0).toInt+r))
              packed.array()
            else raw
          }
          parts(0)->CheckpointIO.Plan("F32",shape,4L*size,read)
        }
        val context=new NativeWorkContext(()=>{check();false},check,(_,_)=>())
        try CheckpointIO.write(out,plans,Map.empty,false,context,"clip_vision_convert")
        catch case _:FileAlreadyExistsException=>()
      }
    require(BrowserFiles.sha256(out)==convertedSha,s"Incorrect converted CLIP weights: $out")
    val path=ujson.write(ujson.Str(out.toAbsolutePath.toString))
    val adapter=ujson.write(ujson.Str(device))
    s"""fn main(input) {
      vision = load_vision_encoder($path, backend=$adapter, params_backend=$adapter)
      return {embedding = encode_vision(vision,input,pooled=true)}
    }"""
  def searchProgram(device:String,check:()=>Unit):String =
    val root=sys.env.get("SDFX_CHARA_CACHE").map(Path.of(_)).getOrElse {
      val home=Path.of(sys.props("user.home"),".cache")
      val old=home.resolve("stablediffusion-nui/chara_searcher")
      val current=home.resolve("stable-diffusion-nui/chara_searcher")
      if !Files.exists(current) && Files.exists(old) then old else current
    }
    program(root,32,device,check)
  def search(raw:ujson.Value,engine:sdfx.vulkan.VulkanInference,device:String,check:()=>Unit):ujson.Value =
    if raw("action").str!="embed" then throw sdfx.core.protocol.EngineError("unsupported","Character dataset import requires the instance detector and segmentation model")
    val vision=searchProgram(device,check)
    val image=DiffusionImageInput.readRgb(Path.of(raw("image_path").str))
    try
      val row=VulkanAesthetic.projection(image,engine,vision,false,false,512,true,false)
      check()
      ujson.Obj("embedding"->ujson.Arr.from(row.map(v=>ujson.Num(v.toDouble))))
    finally engine.unloadClipVision()
