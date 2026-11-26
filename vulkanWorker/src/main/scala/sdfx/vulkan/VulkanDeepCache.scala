package sdfx.vulkan

import ujson.Value

private[vulkan] object VulkanDeepCache:
  val id = "deepcache"

  final case class Config(interval: Int, branch: Int, fullFirst: Int, order: Int, contraction: Double):
    def args: String =
      s", deepcache_interval = $interval, deepcache_branch = $branch, deepcache_first = $fullFirst" +
        s", deepcache_order = $order, deepcache_contraction = $contraction"

  def parse(raw: Value): Config =
    val fields = raw.obj
    val allowed = Set("enabled", "interval", "branch", "full_first", "order", "contraction", "rel_l1_threshold")
    if !fields.keysIterator.forall(allowed) then VulkanProtocol.invalid("Unknown DeepCache option")
    def num(key: String, fallback: Double): Double =
      fields.get(key).fold(fallback)(_.numOpt.getOrElse(Double.NaN))
    def int(key: String, fallback: Int, min: Int, max: Int): Int =
      val value = num(key, fallback)
      if !value.isWhole || value < min || value > max then
        VulkanProtocol.invalid(s"DeepCache $key must be an integer in $min..$max")
      value.toInt
    val contraction = num("contraction", 1)
    if !contraction.isFinite || contraction <= 0 || contraction > 1 then
      VulkanProtocol.invalid("DeepCache contraction must be in (0, 1]")
    val drift = num("rel_l1_threshold", 0)
    if !drift.isFinite || drift < 0 || drift > 2 then
      VulkanProtocol.invalid("DeepCache rel_l1_threshold must be in [0, 2]")
    if drift > 0 then VulkanProtocol.unsupported("DeepCache shallow-feature drift threshold")
    Config(int("interval", 3, 2, 10), int("branch", 0, 0, 3), int("full_first", 5, 0, 50),
      int("order", 0, 0, 2), contraction)
