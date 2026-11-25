// RGB projection adapted from kmizu/scala-torch (BSD-3-Clause). See META-INF/LICENSE-scala-torch.txt.
package sdfx.vulkan

private[vulkan] object VulkanLatentMath:
  def rgb(sdxl: Boolean, r: Double, g: Double, b: Double): Array[Float] =
    val f=if sdxl then Array(Array(.3448,.4168,.4395),Array(-.1953,-.0290,.0250),Array(.1074,.0886,-.0163),Array(-.3730,-.2499,-.2088))
      else Array(Array(.298,.207,.208),Array(.187,.286,.173),Array(-.158,.189,.264),Array(-.184,-.271,-.473))
    val a=Array.ofDim[Double](3,4);val rgb=Array(r,g,b)
    for i <- 0 until 3 do
      for j <- 0 until 3 do a(i)(j)=f.indices.map(k=>f(k)(i)*f(k)(j)).sum+(if i==j then .05 else 0)
      a(i)(3)=rgb(i)
    for i <- 0 until 3 do
      val pivot=(i until 3).maxBy(row=>math.abs(a(row)(i)))
      val swap=a(i);a(i)=a(pivot);a(pivot)=swap
      val scale=a(i)(i)
      for j <- i until 4 do a(i)(j)/=scale
      for row <- 0 until 3 if row!=i do
        val k=a(row)(i)
        for j <- i until 4 do a(row)(j)-=k*a(i)(j)
    f.map(row=>((0 until 3).map(i=>row(i)*a(i)(3)).sum).toFloat)

  def mean(input: String, width: Int, height: Int, label: String, put: (String,String) => String): String =
    val size = width*height
    var length = size
    var flat = put(label+"flat",s"tensor::reshape($input, [$size,4])")
    // Sum pairs on the spatial axis. Preserve an odd tail at every reduction.
    while length > 1 do
      val half = length/2
      def crop(start: Int,count: Int) = s"tensor::crop($flat, axes = [0], start = [$start], size = [$count])"
      val pair = put(s"${label}sum$length",s"tensor::add(${crop(0,half)}, ${crop(half,half)})")
      flat = if length%2==0 then pair else put(s"${label}tail$length",s"tensor::concat([$pair, ${crop(2*half,1)}], axis = 0)")
      length = (length+1)/2
    put(label+"mean",s"tensor::reshape(tensor::div($flat,$size),[1,1,4,1])")
