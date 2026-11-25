package sdfx.vulkan

import java.util.concurrent.ThreadLocalRandom
import sdfx.core.model.GenerationParams
import ujson.Value

private[vulkan] object VulkanHiresTweaks:
  case class Pass(seed: Long, subseed: Long, strength: Double, repeat: Int, width: Int, height: Int, baseSubseed: Long):
    def noise(engine: VulkanInference,w: Int,h: Int): Option[VulkanInference.Latent] =
      Option.when(strength > 0 || width > 0) {
        val sw = if width>0 then width/8 else w
        val sh = if height>0 then height/8 else h
        val noise = engine.noise(sw,sh,seed).values
        val n = sw*sh
        if strength>0 then
          val other = engine.noise(sw,sh,subseed).values
          val dots = new Array[Double](n)
          for i <- 0 until n do
            var aa, bb, ab = 0.0
            for c <- 0 until 4 do
              val a = noise(i+c*n).toDouble
              val b = other(i+c*n).toDouble
              aa += a*a
              bb += b*b
              ab += a*b
            dots(i) = (ab/math.sqrt(aa*bb).max(1e-16)).min(1).max(-1)
          val same = dots.sum/n > .9995
          for i <- 0 until n do
            val angle = math.acos(dots(i))
            val den = math.sin(angle)
            val u = if same then strength else if den.abs < 1e-10 then 1-strength else math.sin((1-strength)*angle)/den
            val v = if same then 1-strength else if den.abs < 1e-10 then strength else math.sin(strength*angle)/den
            for c <- 0 until 4 do
              val j = i+c*n
              noise(j) = (u*noise(j)+v*other(j)).toFloat
        if sw == w && sh == h then new VulkanInference.Latent(w,h,noise)
        else
          val out = engine.noise(w,h,seed)
          val dx = Math.floorDiv(w-sw,2)
          val dy = Math.floorDiv(h-sh,2)
          val copyW = if dx>=0 then sw else sw+2*dx
          val copyH = if dy>=0 then sh else sh+2*dy
          if copyW>0 then
            for c <- 0 until 4; y <- 0 until copyH do
              System.arraycopy(noise,c*n+(y+(-dy).max(0))*sw+(-dx).max(0),out.values,c*w*h+(y+dy.max(0))*w+dx.max(0),copyW)
          out
      }

  def plan(h: Value,p: GenerationParams,base: Long,sub: Long): Vector[(Int,Pass)] =
    def advance(value: Long,offset: Long): Long =
      require(value >= 0 && offset >= 0 && value <= Long.MaxValue-offset,"Hires batch seeds exceed the signed 64-bit range")
      value+offset
    def num(k: String,d: Double) = h.obj.get(k).fold(d)(_.num)
    def integer(k: String,d: Int,max: Int): Int =
      val x = num(k,d)
      require(x.isValidInt && x>=0 && x<=max,s"Invalid hires $k")
      x.toInt
    def seed(k: String): Long =
      val x = h.obj.get(k) match
        case None => 0L
        case Some(ujson.Str(s)) => s.toLongOption.getOrElse(VulkanProtocol.invalid(s"Invalid hires $k"))
        case Some(ujson.Num(n)) if n.isWhole && n.abs <= 9007199254740991L => n.toLong
        case _ => VulkanProtocol.invalid(s"Invalid hires $k")
      require(x >= -1,s"Invalid hires $k")
      if x == -1 then ThreadLocalRandom.current().nextLong(4294967294L) else x
    val count = integer("batch_count",1,16)
    require(count>=1 && p.imagesPerJob*count<=1600,"Invalid hires repeat count")
    val enabled = h.obj.get("seed_enabled").exists(_.bool)
    val extras = enabled && h.obj.get("extras").exists(_.bool)
    val strength = if extras then num("strength",0) else if enabled then 0.0 else p.subseedStrength
    require(strength.isFinite && strength>=0 && strength<=1,"Invalid hires variation strength")
    val w = integer("resize_width",0,16384)
    val ht = integer("resize_height",0,16384)
    require(w%8==0 && ht%8==0,"Hires seed dimensions must be multiples of eight")
    val resized = extras && w>0 && ht>0
    val primary = if enabled then seed("seed") else 0L
    val variation = if extras then seed("subseed") else 0L
    (for iter <- 0 until p.nIter; repeat <- 0 until count; b <- 0 until p.batchSize yield
      val i = iter*p.batchSize+b
      val a = if primary==0 then advance(base,i.toLong) else advance(primary,if strength==0 then i.toLong else 0L)
      val s = advance(if variation==0 then sub else variation,i.toLong+repeat)
      i -> Pass(advance(a,if strength==0 then repeat.toLong else 0L),s,strength,repeat,if resized then w else 0,if resized then ht else 0,advance(sub,i.toLong))
    ).toVector
