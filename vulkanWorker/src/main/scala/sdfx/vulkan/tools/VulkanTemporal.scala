package sdfx.vulkan.tools

import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.nio.file.{Files, LinkOption, Path}
import java.nio.file.attribute.BasicFileAttributes
import java.util.Base64
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters.*
import scala.util.Using
import sdfx.core.png.ScrambledPng
import sdfx.vulkan.TemporalPixels as Pixels
import ujson.{Arr, Obj, Value}

private[vulkan] object VulkanTemporal:
  private val maxFrames = 10000
  private val maxPixels = 2097152L
  private val chars = """\d+|\D+""".r
  private val extensions = Set("png","jpg","jpeg","webp","bmp","tif","tiff")
  private case class Sequence(files: Vector[Path], w: Int, h: Int)
  private def checkedPath(raw: String): Path =
    require(raw.nonEmpty && !Path.of(raw).iterator().asScala.exists(_.toString == ".."),"Invalid Temporal Lab path")
    val p = Path.of(raw).toAbsolutePath.normalize()
    Iterator.iterate(p)(_.getParent).takeWhile(_ != null).foreach { ancestor =>
      if Files.exists(ancestor,LinkOption.NOFOLLOW_LINKS) then
        require(!Files.isSymbolicLink(ancestor) && !Files.readAttributes(ancestor,classOf[BasicFileAttributes],LinkOption.NOFOLLOW_LINKS).isOther,
          s"Linked paths are not supported: $ancestor")
    }
    p
  private def compareNames(a: Path,b: Path): Boolean =
    val first = a.getFileName.toString.toLowerCase(java.util.Locale.ROOT)
    val second = b.getFileName.toString.toLowerCase(java.util.Locale.ROOT)
    val aa = chars.findAllIn(first).toVector
    val bb = chars.findAllIn(second).toVector
    val comparison = aa.zip(bb).iterator.map { (x,y) =>
      if x.head.isDigit && y.head.isDigit then BigInt(x).compare(BigInt(y)) else x.compareTo(y)
    }.find(_ != 0).getOrElse(Integer.compare(aa.size,bb.size))
    if comparison == 0 then first < second else comparison < 0
  private def dimensions(file: Path): (Int,Int) =
    Using.resource(ImageIO.createImageInputStream(file.toFile)) { stream =>
      val readers = ImageIO.getImageReaders(stream)
      require(readers.hasNext,s"Unreadable frame: $file")
      val reader = readers.next()
      try
        reader.setInput(stream)
        val w = reader.getWidth(0)
        val h = reader.getHeight(0)
        require(w > 0 && h > 0 && w <= 16384 && h <= 16384 && w.toLong*h <= maxPixels,"Frame exceeds Temporal Lab working memory limit")
        (w,h)
      finally reader.dispose()
    }
  private def sequence(root: Path, check: () => Unit): Sequence =
    require(Files.isDirectory(root),s"Frame directory missing: $root")
    val paths = Using.resource(Files.list(root)) { entries =>
      val all = entries.limit(20001).iterator().asScala.toVector
      require(all.length <= 20000,"Too many frame directory entries")
      all.filter(p => extensions(p.getFileName.toString.split("\\.").last.toLowerCase(java.util.Locale.ROOT))).sortWith(compareNames)
    }
    require(paths.nonEmpty && paths.size <= maxFrames,"Expect 1..10000 image frames")
    require(paths.map(_.getFileName.toString.toLowerCase(java.util.Locale.ROOT).replaceFirst("\\.[^.]+$","")).distinct.size == paths.size,"Ambiguous duplicate frame names")
    val sizes = paths.map { path =>
      check()
      checkedPath(path.toString)
      require(Files.isRegularFile(path),s"Invalid frame: $path")
      dimensions(path)
    }
    val (w,h) = sizes.head
    require(sizes.forall(_ == ((w,h))) && w.toLong*h*paths.size <= 2000000000L,"Frame sequence dimensions differ or exceed limits")
    Sequence(paths,w,h)
  private def align(image: BufferedImage,w: Int,h: Int): BufferedImage =
    val sw = image.getWidth
    val sh = image.getHeight
    if sw == w && sh == h then image
    else
      val cw = if sw.toLong*h > w.toLong*sh then (sh.toLong*w/h).toInt.max(1) else sw
      val ch = if sw.toLong*h < w.toLong*sh then (sw.toLong*h/w).toInt.max(1) else sh
      DiffusionImageResize.bilinear(image.getSubimage((sw-cw)/2,(sh-ch)/2,cw,ch),w,h)

  def execute(raw: Value, check: () => Unit, publish: () => Unit, progress: (Int,Int,String) => Unit,
      password: Option[String]): Value =
    def text(key: String) = raw(key).str
    def path(key: String) = checkedPath(text(key))
    def num(key: String, default: Double, lo: Double, hi: Double): Double =
      val n = raw.obj.get(key).fold(default)(_.num)
      require(n.isFinite && n >= lo && n <= hi,s"$key must be in $lo..$hi")
      n
    def int(key: String, default: Int, lo: Int, hi: Int, odd: Boolean = false): Int =
      val n = num(key,default,lo,hi)
      require(n.isValidInt && (!odd || n.toInt%2 == 1),s"Invalid $key")
      n.toInt
    val op = text("operation")
    require(Set("preview","dfi","deflicker","fuse","extract")(op),"Unknown Temporal Lab operation")
    val keys = op match
      case "extract" => Seq("video_path")
      case "deflicker" => Seq("frames_dir")
      case "fuse" => Seq("style_dir","video_dir")
      case _ => Seq("reference_dir","generated_dir")
    val inputs = keys.map(path)
    val seqs = if op == "extract" then Nil else inputs.map(sequence(_,check))
    require(seqs.map(_.files.size).distinct.size <= 1,"Frame sequences must have equal lengths")
    var legacyPassword = false
    def read(p: Path): BufferedImage =
      check()
      if p.getFileName.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".png") then
        dimensions(p)
        val decoded = ScrambledPng.read(p,password)
        if decoded.scrambled && decoded.passwordVerified.isEmpty then legacyPassword = true
        decoded.image
      else DiffusionImageInput.readRgb(p,compositeAlpha=false,applyOrientation=false)
    def mask(a: Array[Int],b: Array[Int],w: Int,h: Int) =
      Pixels.mask(a,b,w,h,int("source_denoise",0,0,30),num("tolerance",5,0.5,20),
        int("expand",0,0,50),int("smooth",1,1,99,true),() => check())
    def result(obj: Obj): Obj =
      obj("operation") = op
      if legacyPassword then obj("warnings") = Arr("Legacy image password is unverified; decoded pixels may be incorrect.")
      obj
    if op == "preview" then
      val ref = seqs.head
      val gen = seqs.last
      val idx = int("frame_index",1,1,ref.files.size-1)
      def reference(i: Int) = Pixels.rgb(align(read(ref.files(i)),gen.w,gen.h))
      val samples = mask(reference(idx-1),reference(idx),gen.w,gen.h)
      val gray = Pixels.image(samples.map(_ * 0x010101),gen.w,gen.h)
      val factor = (1024.0 / gen.w.max(gen.h)).min(1)
      val shown = DiffusionImageResize.lanczos(gray,(gen.w*factor).round.toInt.max(1),(gen.h*factor).round.toInt.max(1))
      val bytes = new ByteArrayOutputStream()
      require(ImageIO.write(shown,"png",bytes),"PNG writer unavailable")
      return result(Obj("mask_png_base64" -> Base64.getEncoder.encodeToString(bytes.toByteArray),
        "frame_index" -> idx,"dimensions" -> Arr(gen.w,gen.h),"preview_dimensions" -> Arr(shown.getWidth,shown.getHeight),
        "generated_previous" -> gen.files(idx-1).toString,"generated_current" -> gen.files(idx).toString))

    val parent = path("output_parent").toRealPath()
    val name = text("output_name")
    require(name.nonEmpty && name != "." && name != ".." && !name.exists(c => "/\\:\u0000".contains(c)) &&
      !name.endsWith(" ") && !name.endsWith("."),"Output name must be one safe directory component")
    val dest = parent.resolve(name)
    val realInputs = inputs.map(_.toRealPath())
    require(!realInputs.exists(p => parent.startsWith(p) || p.startsWith(dest)) && !Files.exists(dest,LinkOption.NOFOLLOW_LINKS),"Output exists or overlaps an input")
    val stage = Files.createTempDirectory(parent,".sdfx-temporal-")
    def id(p: Path) = Files.readAttributes(p,classOf[BasicFileAttributes],LinkOption.NOFOLLOW_LINKS).fileKey()
    val identity = (id(parent),id(stage))
    def checkStage(): Unit =
      checkedPath(stage.toString)
      require(identity == ((id(parent),id(stage))),"Output directory identity changed")
    var published = false
    var written = 0
    var totalBytes = 0L
    def write(im: BufferedImage): Unit =
      check()
      val file = stage.resolve(f"frame${written+1}%06d.png")
      password match
        case Some(_) => ScrambledPng.write(im,file,Nil,password)
        case None => require(ImageIO.write(im,"png",file.toFile),"PNG writer unavailable")
      totalBytes += Files.size(file)
      require(totalBytes <= 4L*1024*1024*1024,"Temporal output exceeds 4 GiB")
      written += 1
    try
      val info = if op == "extract" then
        VulkanVideoExtract.run(inputs.head,num("fps",0,0,1000),write,check,(i,n) => progress(i,n,op))
      else
        val source = seqs.head
        val target = seqs.last
        val w = target.w
        val h = target.h
        val info = Obj("dimensions" -> Arr(w,h))
        op match
          case "fuse" =>
            val strength = num("strength",0.5,0.1,1)
            for i <- target.files.indices do
              val style = Pixels.rgb(DiffusionImageResize.lanczos(read(source.files(i)),w,h))
              write(Pixels.image(Pixels.overlay(style,Pixels.rgb(read(target.files(i))),strength,false),w,h))
              progress(i+1,target.files.size,op)
            info("resized_style") = (source.w != w || source.h != h)
          case "deflicker" =>
            val blend = num("blend",0,0,1)
            val overlay = num("overlay",0,0,1)
            val normalize = raw.obj.get("normalize").exists(_.bool)
            def pairs(frames: Iterator[Array[Int]], duplicate: Boolean)(combine: (Array[Int],Array[Int]) => Array[Int]) =
              var previous = frames.next()
              val rest = frames.map { current =>
                val value = combine(previous,current)
                previous = current
                value
              }
              if duplicate && rest.hasNext then
                val first = rest.next()
                Iterator(first,first) ++ rest
              else Iterator.single(previous) ++ rest
            var frames = target.files.iterator.map(p => Pixels.rgb(read(p)))
            if blend > 0 then frames = pairs(frames,false)((a,b) => Pixels.mix(a,b,null,1-blend))
            if overlay > 0 then frames = pairs(frames,true)((a,b) => Pixels.overlay(a,b,overlay,overlay >= 0.4))
            if normalize then frames = pairs(frames,false) { (a,b) =>
              val x = Pixels.meanGray(a)
              val y = Pixels.meanGray(b)
              if x+y == 0 then b else Pixels.mix(a,b,null,x/(x+y))
            }
            frames.foreach { pixels =>
              write(Pixels.image(pixels,w,h))
              progress(written,target.files.size,op)
            }
          case "dfi" =>
            val denoise = int("inter_denoise",9,1,25)
            val sigma = int("inter_denoise_size",9,1,25,true)
            val denoiseEvery = int("inter_denoise_frequency",3,1,15)
            val blur = math.rint(num("inter_blur",3,1,5)).toInt
            val refresh = int("refresh_frequency",0,0,30)
            val preserve = num("preserve",0,0,100)
            val limit = int("frame_limit",0,0,maxFrames)
            val count = if limit==0 then target.files.size else limit.min(target.files.size)
            def reference(i: Int) = Pixels.rgb(align(read(source.files(i)),w,h))
            var prior = reference(0)
            var previous = Pixels.rgb(read(target.files.head))
            write(Pixels.image(previous,w,h))
            progress(1,count,op)
            for i <- 1 until count do
              val ref = reference(i)
              val stable = mask(prior,ref,w,h)
              val percent = math.rint(stable.count(_ == 255)*1000.0/stable.length)/10.0
              val width = math.ceil((100-percent)/3).toInt | 1
              val weights = Pixels.gaussian(stable,w,h,width,false,() => check())
              val retention = if refresh == 0 then percent/100 else if (i-1)%refresh==0 then preserve/100 else 1.0
              previous = Pixels.mix(previous,Pixels.rgb(read(target.files(i))),weights,retention)
              if (i-1)%denoiseEvery==0 then
                if denoise>1 && sigma>1 then previous = Pixels.bilateral(previous,w,h,denoise,sigma,() => check())
                if blur>1 then previous = Pixels.gaussian(previous,w,h,blur|1,true,() => check())
              write(Pixels.image(previous,w,h))
              prior = ref
              progress(i+1,count,op)
            info("reference_aligned") = (source.w != w || source.h != h)
        info
      checkStage()
      check()
      publish()
      Files.move(stage,dest)
      published = true
      info("output_dir") = dest.toString
      info("frame_count") = written
      info("output_bytes") = totalBytes
      result(info)
    finally
      if !published then
        checkStage()
        Using.resource(Files.list(stage))(_.forEach(p => Files.deleteIfExists(p)))
        Files.deleteIfExists(stage)
