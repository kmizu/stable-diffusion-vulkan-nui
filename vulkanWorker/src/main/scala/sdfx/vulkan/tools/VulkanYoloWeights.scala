package sdfx.vulkan.tools

import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}
import java.nio.charset.StandardCharsets
import java.util.zip.ZipFile
import scala.collection.mutable
import scala.jdk.CollectionConverters.*
import scala.util.Using
import ujson.{Arr,Obj}

private[vulkan] object VulkanYoloWeights:
  private val limit = 128L*1024*1024
  private case class Weight(name: String, dtype: String, shape: Vector[Int], bytes: Array[Byte])

  private val release = "53cc19de382014514d9d4038601d261a7faa9b7b"
  private val downloads = Map(
    "face_yolov8n.pt" -> "70b640f8f60b1cf0dcc72f30caf3da9495eb2fb6509da48c53374ad6806e6a9c",
    "face_yolov8s.pt" -> "c7237eff25787377de196961140ceaed324d859ee8de5a775d93d33a0e3fab78",
    "hand_yolov8n.pt" -> "3991202eb69e9ddcb3b9ba80cdeb41e734ffaf844403d6c9f47d515cd88c6f29")

  def resolve(name: String, modelDir: Path, check: () => Unit): Path =
    val direct = Path.of(name)
    val hf = sys.env.get("HF_HUB_CACHE").map(Path.of(_)).getOrElse(
      sys.env.get("HF_HOME").map(Path.of(_)).getOrElse(Path.of(sys.props("user.home"),".cache","huggingface")).resolve("hub"))
    val catalog = hf.resolve("models--Bingsu--adetailer")
    val ref = catalog.resolve("refs/main")
    val revision = Option.when(Files.isRegularFile(ref))(Files.readString(ref).trim).filter(_.matches("[0-9a-f]{40}"))
    val path = (Vector(direct,modelDir.resolve("adetailer").resolve(direct)) ++ revision.map(r => catalog.resolve("snapshots").resolve(r).resolve(direct)))
      .find(Files.isRegularFile(_)).getOrElse {
        val sha=downloads.getOrElse(name,throw IllegalArgumentException(s"ADetailer detection weights unavailable: $name"))
        val directory=Files.createDirectories(Path.of(sys.props("user.home"),".cache","sdv-nui","detectors"))
        VulkanClip.asset(directory,name,s"https://huggingface.co/Bingsu/adetailer/resolve/$release/$name",sha,check,(_,_,_) => ())
      }
    if name.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") then path.toAbsolutePath
    else
      require(name.toLowerCase(java.util.Locale.ROOT).endsWith(".pt"),"Choose YOLOv8 detection .pt or safetensors weights")
      require(Files.size(path) <= limit,"YOLO detection checkpoint exceeds 128 MiB")
      val hash = sdfx.core.imagebrowser.BrowserFiles.sha256(path)
      val cache = Files.createDirectories(Path.of(sys.props("user.home"),".cache","sdv-nui","yolov8-v1"))
      val target = cache.resolve(hash+".safetensors")
      if !Files.isRegularFile(target) then
        val temp = Files.createTempFile(cache,"convert-",".part")
        try
          convert(path,temp)
          try Files.move(temp,target) catch case _: java.nio.file.FileAlreadyExistsException => ()
        finally Files.deleteIfExists(temp)
      target

  private def convert(path: Path, dest: Path): Unit = Using.resource(new ZipFile(path.toFile)) { zip =>
    val entries = zip.entries.asScala.toVector
    require(entries.size < 12000 && entries.forall(e => e.getSize >= 0 && e.getSize <= limit) && entries.map(_.getSize).sum <= limit && entries.map(_.getName).distinct.size == entries.size,"Invalid YOLO archive size")
    val pickle = entries.find(_.getName.endsWith("/data.pkl")).getOrElse(throw IllegalArgumentException("Missing YOLO state dictionary"))
    require(pickle.getSize <= 16000000,"Oversized YOLO state dictionary")
    val root = pickle.getName.stripSuffix("data.pkl")
    def bytes(entry: java.util.zip.ZipEntry): Array[Byte] = Using.resource(zip.getInputStream(entry))(_.readNBytes(Math.toIntExact(entry.getSize)))
    val net = TorchYoloPickle.model(bytes(pickle))
    val head = net.children.get("model").asInstanceOf[TorchYoloPickle.Module].children.get("22").asInstanceOf[TorchYoloPickle.Module]
    require(head.`type`.endsWith(".Detect"),"Only YOLOv8 detection models are supported")
    val params = mutable.LinkedHashMap.empty[String,TorchYoloPickle.Tensor]
    val eps = mutable.Map.empty[String,Double]
    var visited = 0
    def walk(module: TorchYoloPickle.Module, prefix: String, depth: Int): Unit =
      visited += 1
      require(visited < 6000 && depth < 30 && params.size<6000,"Invalid YOLO module graph")
      for field <- Seq("_parameters","_buffers") do
        TorchYoloPickle.map(module.state.get(field)).asScala.foreach {
          case (k:String,t:TorchYoloPickle.Tensor) => params(prefix+k)=t
          case _ => ()
        }
      if module.`type`.endsWith("BatchNorm2d") then eps(prefix) = module.state.get("eps").asInstanceOf[Number].doubleValue
      module.children.asScala.foreach { (key,child) => if child!=null then walk(child.asInstanceOf[TorchYoloPickle.Module],prefix+key.toString+".",depth+1) }
    walk(net,"",0)
    val storage = mutable.Map.empty[String,Array[Byte]]
    def shape(t: TorchYoloPickle.Tensor): Vector[Int] = t.shape.asScala.toVector.map(_.intValue)
    def values(t: TorchYoloPickle.Tensor): Array[Float] =
      val dims=shape(t)
      require(dims.nonEmpty && dims.forall(_>0),"Invalid YOLO tensor shape")
      val total = dims.foldLeft(1)(Math.multiplyExact)
      val stride = dims.scanRight(1)(Math.multiplyExact).tail
      require(t.stride.asScala.map(_.intValue).toVector == stride && t.offset >= 0,"Unsupported YOLO tensor layout")
      val raw=storage.getOrElseUpdate(t.storage,bytes(Option(zip.getEntry(root+"data/"+t.storage)).getOrElse(throw IllegalArgumentException("Missing YOLO storage"))))
      val width = t.dtype match
        case "torch.FloatStorage" => 4
        case "torch.HalfStorage" => 2
        case other => throw IllegalArgumentException(s"Unsupported YOLO weights: $other")
      require((t.offset.toLong+total)*width <= raw.length,"Invalid YOLO storage bounds")
      val buffer=ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
      buffer.position(t.offset*width)
      Array.fill(total)(if width==2 then java.lang.Float.float16ToFloat(buffer.getShort()) else buffer.getFloat())
    val output = mutable.ArrayBuffer.empty[Weight]
    def add(name:String,dims:Vector[Int],v:Array[Float],half:Boolean): Unit =
      val buffer=ByteBuffer.allocate(v.length*(if half then 2 else 4)).order(ByteOrder.LITTLE_ENDIAN)
      if half then v.foreach(n=>buffer.putShort(java.lang.Float.floatToFloat16(n))) else v.foreach(buffer.putFloat)
      output += Weight(name,if half then "F16" else "F32",dims,buffer.array)
    for (key,t) <- params if key.startsWith("model.") && key.endsWith(".weight") && !key.contains(".bn.") && !key.endsWith("dfl.conv.weight") do
      val dims=shape(t)
      val w=values(t)
      val base=key.stripSuffix("weight")
      val bias=params.get(base+"bias").map(values)
      val bn=base.stripSuffix("conv.")+"bn."
      if base.endsWith("conv.") && eps.contains(bn) then
        val gamma=values(params(bn+"weight"));val beta=values(params(bn+"bias"))
        val mean=values(params(bn+"running_mean"));val variance=values(params(bn+"running_var"))
        val plane=w.length/dims.head
        val result=Array.tabulate(dims.head) { c =>
          val factor=(gamma(c)/math.sqrt(variance(c)+eps(bn))).toFloat
          var i=c*plane
          while i < (c+1)*plane do
            w(i)*=factor
            i+=1
          beta(c)+(bias.fold(0f)(_(c))-mean(c))*factor
        }
        add(base+"bias",Vector(dims.head),result,false)
      else bias.foreach(a => add(base+"bias",Vector(a.length),a,false))
      add(key,dims,w,true)
    require(Seq("model.0.conv.weight","model.22.cv2.0.2.weight","model.22.cv3.0.2.weight").forall(k=>output.exists(_.name==k)),"Unsupported YOLOv8 detection layout")
    val names=Obj.from(TorchYoloPickle.map(net.state.get("names")).asScala.iterator.map((k,v)=>k.toString->ujson.Str(v.toString)))
    val metadata=Obj("format"->"pt","yolov8.variant"->"detect","yolov8.input_size"->"640",
      "yolov8.num_classes"->head.state.get("nc").toString,"yolov8.reg_max"->head.state.get("reg_max").toString,"yolov8.names"->names.render())
    val header=Obj("__metadata__"->metadata)
    var offset=0
    output.foreach { w =>
      header(w.name)=Obj("dtype"->w.dtype,"shape"->Arr.from(w.shape),"data_offsets"->Arr(offset,offset+w.bytes.length))
      offset+=w.bytes.length
    }
    val json=header.render().getBytes(StandardCharsets.UTF_8)
    val padding=(8-json.length%8)%8
    Using.resource(Files.newOutputStream(dest)) { out =>
      out.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(json.length+padding).array)
      out.write(json);out.write(Array.fill[Byte](padding)(32));output.foreach(w=>out.write(w.bytes))
    }
  }
