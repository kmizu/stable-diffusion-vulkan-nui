package sdfx.ui

import munit.FunSuite
import sdfx.ui.view.ZoomImageView

class ImageLightboxSuite extends FunSuite:
  test("hitTestImage accepts points inside image rectangle including boundary") {
    assert(ZoomImageView.hitTestImage(10, 20, 10, 20, 100, 80))
    assert(ZoomImageView.hitTestImage(60, 60, 10, 20, 100, 80))
    assert(ZoomImageView.hitTestImage(110, 100, 10, 20, 100, 80))
    assert(!ZoomImageView.hitTestImage(9.9, 60, 10, 20, 100, 80))
    assert(!ZoomImageView.hitTestImage(110.1, 60, 10, 20, 100, 80))
    assert(!ZoomImageView.hitTestImage(60, 19.9, 10, 20, 100, 80))
    assert(!ZoomImageView.hitTestImage(60, 100.1, 10, 20, 100, 80))
    assert(!ZoomImageView.hitTestImage(0, 0, 0, 0, 0, 100))
    assert(!ZoomImageView.hitTestImage(0, 0, 0, 0, 100, 0))
  }

  test("clampZoom bounds zoom ratio to supported range") {
    assertEquals(ZoomImageView.clampZoom(0.01), 0.05)
    assertEquals(ZoomImageView.clampZoom(1.5), 1.5)
    assertEquals(ZoomImageView.clampZoom(42.0), 16.0)
  }

  test("formatMeta formats dimensions and zoom label cleanly") {
    assertEquals(ZoomImageView.formatMeta(0, 512, None), "")
    assertEquals(ZoomImageView.formatMeta(512, 0, Some(1.0)), "")
    assertEquals(ZoomImageView.formatMeta(1024, 768, None), "1024 × 768 · Fit")
    assertEquals(ZoomImageView.formatMeta(1024, 768, Some(1.0)), "1024 × 768 · 100%")
    assertEquals(ZoomImageView.formatMeta(512, 512, Some(2.5)), "512 × 512 · 250%")
  }
