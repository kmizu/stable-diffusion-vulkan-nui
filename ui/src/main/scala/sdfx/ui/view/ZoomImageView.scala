package sdfx.ui.view

import javafx.scene.Cursor
import javafx.scene.image.{Image, ImageView}
import javafx.scene.input.{MouseButton, ScrollEvent}
import javafx.scene.layout.Region
import javafx.scene.shape.Rectangle
import sdfx.ui.vm.MaskGeometry

/** Image viewer: fit by default, wheel/Ctrl+wheel zoom around the cursor, drag to pan, double click = fit / 100%. */
final class ZoomImageView extends Region:
  private val view = new ImageView()
  view.setSmooth(true)
  getChildren.add(view)
  getStyleClass.add("zoom-view")
  private val clip = new Rectangle()
  clip.widthProperty.bind(widthProperty)
  clip.heightProperty.bind(heightProperty)
  setClip(clip)

  private var zoom: Option[Double] = None // None = fit
  private var ox = 0.0
  private var oy = 0.0
  private var dragFrom: Option[(Double, Double, Double, Double)] = None

  /** When true, scrolling the mouse wheel zooms without holding Ctrl (used inside the fullscreen lightbox). */
  var wheelZoomWithoutCtrl: Boolean = false
  /** Fired on a single primary click directly on the fitted image. */
  var onImageClick: Option[() => Unit] = None
  /** Fired on a single primary click outside the rendered image bounds. */
  var onBackdropClick: Option[() => Unit] = None
  /** Fired whenever the image or zoom level changes. */
  var onZoomChanged: Option[() => Unit] = None

  def image: Image = view.getImage
  def isZoomed: Boolean = zoom.isDefined
  def zoomFactor: Option[Double] = zoom

  private var lastSize: Option[(Double, Double)] = None

  /** Shows an image. Until a background image is loaded the view is hidden (no stretched stale fit);
    * the zoom is kept only when the loaded image has the same size as the previous one.
    */
  def setImage(img: Image): Unit =
    view.setImage(img)
    if img == null then
      zoom = None
      lastSize = None
      requestLayout()
      onZoomChanged.foreach(_())
    else
      view.setVisible(false)
      ImageFiles.whenLoaded(img) {
        if view.getImage eq img then
          val size = (img.getWidth, img.getHeight)
          if !lastSize.contains(size) then zoom = None
          lastSize = Some(size)
          view.setVisible(!img.isError)
          requestLayout()
          onZoomChanged.foreach(_())
      }

  def fit(): Unit =
    zoom = None
    requestLayout()
    onZoomChanged.foreach(_())

  def setZoom100(): Unit =
    if view.getImage != null then zoomAt(getWidth / 2, getHeight / 2, 1.0)

  def zoomBy(factor: Double): Unit =
    if view.getImage != null then zoomAt(getWidth / 2, getHeight / 2, currentScale * factor)

  private def imgSize: (Double, Double) =
    val i = view.getImage
    if i == null then (0.0, 0.0) else (i.getWidth, i.getHeight)

  private def currentScale: Double =
    val (iw, ih) = imgSize
    zoom.getOrElse(MaskGeometry.fit(iw, ih, getWidth, getHeight).scale)

  def isPointOnImage(x: Double, y: Double): Boolean =
    val i = view.getImage
    i != null && !i.isError && ZoomImageView.hitTestImage(x, y, ox, oy, view.getFitWidth, view.getFitHeight)

  override protected def layoutChildren(): Unit =
    val (iw, ih) = imgSize
    if iw > 0 && ih > 0 then
      zoom match
        case None =>
          val f = MaskGeometry.fit(iw, ih, getWidth, getHeight)
          ox = f.offsetX
          oy = f.offsetY
          view.setFitWidth(f.width)
          view.setFitHeight(f.height)
        case Some(z) =>
          view.setFitWidth(iw * z)
          view.setFitHeight(ih * z)
          // keep the zoomed image reachable after resizes: center a dimension that fits, else clamp the pan
          ox = clampOffset(ox, iw * z, getWidth)
          oy = clampOffset(oy, ih * z, getHeight)
      view.relocate(ox, oy)

  private def clampOffset(o: Double, size: Double, box: Double): Double =
    if size <= box then (box - size) / 2 else o.min(0).max(box - size)

  private def zoomAt(x: Double, y: Double, z: Double): Unit =
    val s = currentScale
    val nz = ZoomImageView.clampZoom(z)
    val ix = (x - ox) / s
    val iy = (y - oy) / s
    ox = x - ix * nz
    oy = y - iy * nz
    zoom = Some(nz)
    requestLayout()
    onZoomChanged.foreach(_())

  private def updateHoverCursor(x: Double, y: Double): Unit =
    if dragFrom.isDefined then setCursor(Cursor.CLOSED_HAND)
    else if view.getImage != null && isPointOnImage(x, y) then
      if zoom.isDefined then setCursor(Cursor.OPEN_HAND)
      else if onImageClick.isDefined then setCursor(Cursor.HAND)
      else setCursor(Cursor.DEFAULT)
    else setCursor(Cursor.DEFAULT)

  addEventFilter(ScrollEvent.SCROLL, (e: ScrollEvent) =>
    if (wheelZoomWithoutCtrl || e.isShortcutDown) && view.getImage != null then
      val f = if e.getDeltaY > 0 then 1.2 else 1 / 1.2
      zoomAt(e.getX, e.getY, currentScale * f)
      e.consume()
  )
  setOnMouseMoved(e => updateHoverCursor(e.getX, e.getY))
  setOnMouseExited(_ => if dragFrom.isEmpty then setCursor(Cursor.DEFAULT))
  setOnMousePressed(e =>
    if e.getButton == MouseButton.PRIMARY && zoom.isDefined then
      dragFrom = Some((e.getX, e.getY, ox, oy))
      setCursor(Cursor.CLOSED_HAND)
  )
  setOnMouseDragged(e =>
    dragFrom.foreach { (sx, sy, sox, soy) =>
      ox = sox + e.getX - sx
      oy = soy + e.getY - sy
      requestLayout()
    }
  )
  setOnMouseReleased(e =>
    dragFrom = None
    updateHoverCursor(e.getX, e.getY)
  )
  setOnMouseClicked(e =>
    if e.getButton == MouseButton.PRIMARY && e.isStillSincePress then
      if e.getClickCount == 1 then
        if view.getImage != null && isPointOnImage(e.getX, e.getY) && zoom.isEmpty && onImageClick.isDefined then
          onImageClick.foreach(_())
          e.consume()
        else if onBackdropClick.isDefined && (view.getImage == null || !isPointOnImage(e.getX, e.getY)) then
          onBackdropClick.foreach(_())
          e.consume()
      else if e.getClickCount == 2 && view.getImage != null && (onImageClick.isEmpty || zoom.isDefined) then
        if zoom.isDefined then fit() else zoomAt(e.getX, e.getY, 1.0)
        e.consume()
  )

object ZoomImageView:
  def hitTestImage(x: Double, y: Double, ox: Double, oy: Double, width: Double, height: Double): Boolean =
    width > 0 && height > 0 && x >= ox && x <= ox + width && y >= oy && y <= oy + height

  def clampZoom(z: Double): Double =
    z.max(0.05).min(16.0)

  def formatMeta(imgW: Double, imgH: Double, zoom: Option[Double]): String =
    if imgW <= 0 || imgH <= 0 then ""
    else s"${imgW.round} × ${imgH.round} · ${zoom.fold("Fit")(z => s"${(z * 100).round}%")}"
