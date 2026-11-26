package sdfx.vulkan

import java.nio.{ByteBuffer, ByteOrder}
import java.util.List
import sdfx.vulkan.tools.{TorchYoloPickle, VulkanFooocusFiles}

class VulkanFooocusSuite extends munit.FunSuite:
  test("dequantizes an offset byte tensor with signed bounds") {
    def tensor(dtype: String, offset: Int, shape: List[Integer], stride: List[Integer]) =
      new TorchYoloPickle.Tensor("0", dtype, offset, shape, stride)
    val quantized = tensor("torch.ByteStorage", 1, List.of(Integer.valueOf(3)), List.of(Integer.valueOf(2)))
    val bound = tensor("torch.HalfStorage", 1, List.of[Integer](), List.of[Integer]())
    def half(value: Float) =
      ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
        .putShort(0.toShort).putShort(java.lang.Float.floatToFloat16(value)).array()
    val values = VulkanFooocusFiles.dequantize(
      quantized, Array[Byte](27, 0, 95, 127, 95, -1, 25), bound, half(-2f), bound, half(2f), ByteOrder.LITTLE_ENDIAN)
    assertEquals(values.toSeq, Seq(-2f, 127f / 255f * 4f - 2f, 2f))
  }

  test("broadcasts per-output-channel and spatial quantization bounds") {
    val weight = new TorchYoloPickle.Tensor("0", "torch.ByteStorage", 0, List.of(2, 2, 2), List.of(4, 2, 1))
    val bound = new TorchYoloPickle.Tensor("1", "torch.FloatStorage", 0, List.of(2, 1, 2), List.of(2, 2, 1))
    def floats(xs: Float*) =
      val b = ByteBuffer.allocate(4 * xs.size).order(ByteOrder.LITTLE_ENDIAN)
      xs.foreach(b.putFloat(_))
      b.array()
    val values = VulkanFooocusFiles.dequantize(weight, Array[Byte](0, -1, -1, 0, 0, -1, -1, 0),
      bound, floats(-2f, -3f, -4f, -5f), bound, floats(2f, 3f, 4f, 5f), ByteOrder.LITTLE_ENDIAN)
    assertEquals(values.toSeq, Seq(-2f, 3f, 2f, -3f, -4f, 5f, 4f, -5f))
  }

  test("reads the floating-point deltas stored with identity bounds in v2.6") {
    val weight = new TorchYoloPickle.Tensor("0", "torch.FloatStorage", 0, List.of(2), List.of(1))
    val bound = new TorchYoloPickle.Tensor("1", "torch.HalfStorage", 0, List.of(1), List.of(1))
    val weights = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putFloat(-1f).putFloat(.5f).array()
    val maximum = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN)
      .putShort(java.lang.Float.floatToFloat16(255f)).array()
    val values = VulkanFooocusFiles.dequantize(weight, weights, bound, Array[Byte](0, 0),
      bound, maximum, ByteOrder.LITTLE_ENDIAN)
    assertEquals(values.toSeq, Seq(-1f, .5f))
  }
