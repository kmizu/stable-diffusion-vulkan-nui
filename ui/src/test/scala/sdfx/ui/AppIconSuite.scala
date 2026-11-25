package sdfx.ui

import munit.FunSuite
import sdfx.ui.view.AppIcon

import java.io.ByteArrayInputStream
import javax.imageio.ImageIO

class AppIconSuite extends FunSuite:
  test("every declared app icon PNG resource exists with expected dimensions") {
    assert(AppIcon.Sizes.nonEmpty)
    AppIcon.Sizes.foreach { size =>
      val path = AppIcon.resourcePath(size)
      val stream = getClass.getResourceAsStream(path)
      assert(stream != null, s"missing icon resource: $path")
      try
        val img = ImageIO.read(stream)
        assert(img != null, s"could not decode PNG: $path")
        assertEquals(img.getWidth, size, s"width mismatch for $path")
        assertEquals(img.getHeight, size, s"height mismatch for $path")
      finally stream.close()
    }
  }

  test("Windows .ico resource exists with valid ICO header") {
    val stream = getClass.getResourceAsStream("/sdfx/ui/icons/sdnui.ico")
    assert(stream != null, "missing /sdfx/ui/icons/sdnui.ico")
    val bytes =
      try stream.readAllBytes()
      finally stream.close()
    assert(bytes.length > 22, "sdnui.ico is too small")
    assertEquals(bytes(0).toInt & 0xff, 0)
    assertEquals(bytes(1).toInt & 0xff, 0)
    assertEquals(bytes(2).toInt & 0xff, 1)
    assertEquals(bytes(3).toInt & 0xff, 0)
    val count = (bytes(4).toInt & 0xff) | ((bytes(5).toInt & 0xff) << 8)
    assertEquals(count, AppIcon.Sizes.size)
  }
