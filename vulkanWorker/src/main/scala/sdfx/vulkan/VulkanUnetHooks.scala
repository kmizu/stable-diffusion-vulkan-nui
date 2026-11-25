package sdfx.vulkan

import sdfx.core.model.GenerationParams
import ujson.Value

private[vulkan] object VulkanUnetHooks:
  val hooks = Set("freeu","hypertile","pag","sag")

  final case class Settings(args: String, regions: Int, pagScale: Double, adaptive: Double, sagScale: Double):
    def guidance(cfg: Double): String =
      if pagScale == 0 then s"guided = cfg(pos, neg, scale = $cfg, sag_scale = $sagScale)"
      else
        val g = math.max(1.0,cfg)
        val guided = if cfg > 1 then s"cfg(enhanced,neg,scale=$cfg)" else "enhanced"
        s"perturbed = predictor(unet,latent_pos$args,pag_regions=$regions)\n" +
          s"enhanced = cfg(pos,perturbed,scale=${1+pagScale/g},pag_adaptive_decay=${adaptive/g})\n" +
          s"guided = $guided"

  def parse(p: GenerationParams, xl: Boolean, seed: Long): Settings =
    def enabled(k: String): Option[Value] = p.hooks.get(k).filterNot(_.obj.get("enabled").contains(ujson.Bool(false)))
    def num(o: Value, k: String, default: Double, min: Double, max: Double): Double =
      val v = o.obj.get(k).fold(default)(_.num)
      require(v >= min && v <= max, s"Invalid $k")
      v
    def int(o: Value, k: String, default: Int, min: Int, max: Int): Int =
      val v = num(o,k,default,min,max)
      require(v.isWhole,s"$k must be an integer")
      v.toInt
    val args = new StringBuilder
    enabled("freeu").foreach { o =>
      Seq(("b1",if xl then 1.3 else 1.5,2),("b2",if xl then 1.4 else 1.6,2),("s1",0.9,4),("s2",0.2,4))
        .foreach { (k,d,max) => args ++= s",freeu_$k=${num(o,k,d,0,max)}" }
    }
    enabled("hypertile").foreach { o =>
      args ++= s",hypertile_size=${int(o,"tile_size",256,32,65536)},hypertile_swap=${int(o,"swap_size",2,1,256)}"
      args ++= s",hypertile_depth=${int(o,"max_depth",0,0,10)},hypertile_scale_depth=${o.obj.get("scale_depth").exists(_.bool)}"
      args ++= s",hypertile_seed=${(seed ^ (seed >>> 32)) & Int.MaxValue}"
    }
    val sag = enabled("sag").fold(0.0)(o => num(o,"scale",0.5,0,30))
    enabled("pag") match
      case None => Settings(args.result(),0,0,0,sag)
      case Some(o) =>
        val stages = if xl then 3 else 4
        val layer = """(down|up)\.(?:block_)?([0-3])""".r
        val names = o.obj.get("layers").fold(Seq("mid"))(_.arr.toSeq.map(_.str.trim))
        require(names.nonEmpty,"PAG requires at least one U-Net layer")
        val regions = names.foldLeft(0) { (bits, name) =>
          val value = name match
            case "mid" | "mid_block" => 256
            case "up" => ((1 << stages)-1) << 4
            case "down" => (1 << stages)-1
            case layer(region,index) if index.toInt < stages => 1 << (index.toInt + (if region=="up" then 4 else 0))
            case _ => VulkanProtocol.unsupported(s"PAG layer $name. Use mid, down, up, down.block_N or up.block_N")
          bits | value
        }
        Settings(args.result(),regions,num(o,"scale",3,0,30),num(o,"adaptive_scale",0,0,1),sag)