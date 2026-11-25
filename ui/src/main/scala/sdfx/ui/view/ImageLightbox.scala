package sdfx.ui.view

import javafx.geometry.{Insets, Pos}
import javafx.scene.control.Button
import javafx.scene.image.Image
import javafx.scene.layout.{HBox, Priority, StackPane}
import sdfx.ui.Fx

import java.nio.file.Path

/** Full-window click-to-enlarge modal viewer (A1111-style lightbox) with zoom/pan, copy, folder reveal, and gallery navigation. */
final class ImageLightbox(
    ctx: UiContext,
    onNavigate: Int => Option[Path],
    canNavigate: () => Boolean
):
  private val zoom = new ZoomImageView()
  zoom.setId("lightbox-zoom")
  zoom.wheelZoomWithoutCtrl = true
  zoom.onBackdropClick = Some(() => close())
  zoom.onZoomChanged = Some(() => refreshHud())

  private val imageReads = new sdfx.ui.vm.ImageReadGate[ImageFiles.DisplayImage](
    ctx.services.imageScrambleSession,
    run => Fx.run(run()),
    (path, secret) => ImageFiles.readForDisplay(path, secret)
  )(using Fx.bg)

  private var currentPath: Option[Path] = None
  private var allowGalleryNav: Boolean = true

  private val titleLabel = W.label("", "lightbox-title")
  private val metaLabel = W.label("", "lightbox-meta")
  private val statusLabel = W.label("", "lightbox-status")
  statusLabel.setMouseTransparent(true)
  statusLabel.setVisible(false)

  private val zoomOutBtn = lightboxBtn("-", "Zoom out (-)", null)(zoomOut())
  private val zoomResetBtn = lightboxBtn("100%", "Actual size (100%)", null)(toggleActualSize())
  private val zoomInBtn = lightboxBtn("+", "Zoom in (+)", null)(zoomIn())
  private val copyBtn = lightboxBtn("", "Copy image", Icons.copy(14))(currentPath.foreach(p => ImageActions.copyImage(ctx, p)))
  private val folderBtn = lightboxBtn("", "Show in folder", Icons.folder(14))(currentPath.foreach(ImageActions.openFolder))
  private val closeBtn = lightboxBtn("", "Close (Esc)", Icons.close(14))(close())
  closeBtn.setId("lightbox-close-btn")

  private def lightboxBtn(text: String, tip: String, icon: javafx.scene.Node)(f: => Unit): Button =
    val b = W.button(text, tip, Seq("lightbox-btn"))(f)
    if icon != null then b.setGraphic(icon)
    b

  private val topBar = new HBox(
    10,
    titleLabel,
    metaLabel,
    W.spacer(),
    zoomOutBtn,
    zoomResetBtn,
    zoomInBtn,
    copyBtn,
    folderBtn,
    closeBtn
  )
  topBar.setAlignment(Pos.CENTER_LEFT)
  topBar.getStyleClass.add("lightbox-topbar")
  topBar.setMaxWidth(Double.MaxValue)
  topBar.setMaxHeight(javafx.scene.layout.Region.USE_PREF_SIZE)
  HBox.setHgrow(titleLabel, Priority.NEVER)

  private val prevBtn = W.vectorButton(Icons.chevronLeft(20), "Previous image (Left)")(navigate(-1))
  prevBtn.getStyleClass.add("lightbox-nav-btn")
  prevBtn.setId("lightbox-prev-btn")
  private val nextBtn = W.vectorButton(Icons.chevronRight(20), "Next image (Right)")(navigate(1))
  nextBtn.getStyleClass.add("lightbox-nav-btn")
  nextBtn.setId("lightbox-next-btn")

  val node: StackPane = new StackPane(zoom, statusLabel, prevBtn, nextBtn, topBar)
  node.setId("image-lightbox")
  node.getStyleClass.add("lightbox-overlay")
  node.setVisible(false)
  node.setManaged(false)
  StackPane.setAlignment(topBar, Pos.TOP_CENTER)
  StackPane.setAlignment(prevBtn, Pos.CENTER_LEFT)
  StackPane.setMargin(prevBtn, new Insets(0, 0, 0, 18))
  StackPane.setAlignment(nextBtn, Pos.CENTER_RIGHT)
  StackPane.setMargin(nextBtn, new Insets(0, 18, 0, 0))
  StackPane.setAlignment(statusLabel, Pos.CENTER)

  ctx.watchImageAccess(node) { () =>
    if isShowing then currentPath.foreach(p => openPath(p, allowGalleryNav = allowGalleryNav))
  }

  def isShowing: Boolean = node.isVisible

  private def showOverlay(): Unit =
    node.setManaged(true)
    node.setVisible(true)
    node.toFront()
    refreshNavButtons()

  private def refreshNavButtons(): Unit =
    val nav = allowGalleryNav && canNavigate()
    prevBtn.setVisible(nav)
    prevBtn.setManaged(nav)
    nextBtn.setVisible(nav)
    nextBtn.setManaged(nav)

  private def refreshHud(): Unit =
    val img = zoom.image
    val meta =
      if img != null && !img.isError then ZoomImageView.formatMeta(img.getWidth, img.getHeight, zoom.zoomFactor)
      else ""
    metaLabel.setText(meta)
    if zoom.isZoomed then
      zoomResetBtn.setText("Fit")
      W.tip(zoomResetBtn, "Fit to window")
    else
      zoomResetBtn.setText("100%")
      W.tip(zoomResetBtn, "Actual size (100%)")

  private def toggleActualSize(): Unit =
    if zoom.isZoomed then zoom.fit() else zoom.setZoom100()

  def openPath(path: Path, title: Option[String] = None, allowGalleryNav: Boolean = true): Unit =
    this.allowGalleryNav = allowGalleryNav
    currentPath = Some(path)
    zoom.setUserData(path)
    zoom.setImage(null)
    titleLabel.setText(title.getOrElse(path.getFileName.toString))
    metaLabel.setText("")
    statusLabel.setText("Loading image…")
    statusLabel.setVisible(true)
    copyBtn.setVisible(true)
    copyBtn.setManaged(true)
    folderBtn.setVisible(true)
    folderBtn.setManaged(true)
    showOverlay()
    imageReads.request(path) {
      case scala.util.Success(display) if isShowing && zoom.getUserData == path =>
        statusLabel.setVisible(false)
        zoom.setImage(display.image)
      case scala.util.Failure(error) if isShowing && zoom.getUserData == path =>
        zoom.setImage(null)
        statusLabel.setText("Cannot display image: " + Fx.message(error).stripPrefix("requirement failed: "))
        statusLabel.setVisible(true)
      case _ => ()
    }

  def openImage(image: Image, title: String = "Preview"): Unit =
    allowGalleryNav = false
    currentPath = None
    imageReads.invalidate()
    zoom.setUserData(null)
    titleLabel.setText(title)
    statusLabel.setVisible(false)
    copyBtn.setVisible(false)
    copyBtn.setManaged(false)
    folderBtn.setVisible(false)
    folderBtn.setManaged(false)
    zoom.setImage(image)
    showOverlay()
    refreshHud()

  def navigate(delta: Int): Unit =
    if isShowing && allowGalleryNav then
      onNavigate(delta).foreach { p =>
        if !currentPath.contains(p) then openPath(p, allowGalleryNav = true)
      }

  def zoomIn(): Unit =
    if isShowing then zoom.zoomBy(1.25)

  def zoomOut(): Unit =
    if isShowing then zoom.zoomBy(1.0 / 1.25)

  def resetZoom(): Unit =
    if isShowing then zoom.fit()

  def close(): Unit =
    imageReads.invalidate()
    currentPath = None
    zoom.setUserData(null)
    zoom.setImage(null)
    statusLabel.setVisible(false)
    node.setVisible(false)
    node.setManaged(false)
