package sdfx.ui.view

import javafx.geometry.Pos
import javafx.scene.Group
import javafx.scene.Node
import javafx.scene.layout.StackPane
import javafx.scene.shape.{SVGPath, StrokeLineCap, StrokeLineJoin}

/** Crisp 24x24-grid vector icons scaled cleanly for native desktop controls. */
object Icons:
  private def svg(d: String, filled: Boolean = false, extraClass: String = ""): SVGPath =
    val p = new SVGPath()
    p.setContent(d)
    if filled then p.getStyleClass.add("svg-icon-fill")
    else
      p.getStyleClass.add("svg-icon")
      p.setStrokeLineCap(StrokeLineCap.ROUND)
      p.setStrokeLineJoin(StrokeLineJoin.ROUND)
    if extraClass.nonEmpty then p.getStyleClass.add(extraClass)
    p

  def icon(size: Double, paths: SVGPath*): StackPane =
    val group = new Group(paths*)
    val scale = size / 24.0
    group.setScaleX(scale)
    group.setScaleY(scale)
    val box = new StackPane(group)
    box.setAlignment(Pos.CENTER)
    box.setPrefSize(size, size)
    box.setMinSize(size, size)
    box.setMaxSize(size, size)
    box.setMouseTransparent(true)
    box.getStyleClass.add("svg-icon-box")
    box

  def path(d: String, size: Double = 15, filled: Boolean = false, extraClass: String = ""): StackPane =
    icon(size, svg(d, filled, extraClass))

  def play(size: Double = 14): StackPane =
    path("M7 5.5v13l11-6.5z", size, filled = true)

  def stop(size: Double = 13): StackPane =
    path("M6 6h12v12H6z", size, filled = true)

  def plus(size: Double = 14): StackPane =
    path("M12 5v14M5 12h14", size)

  def close(size: Double = 13): StackPane =
    path("M6 6l12 12M18 6L6 18", size)

  def refresh(size: Double = 15): StackPane =
    path("M21 12a9 9 0 1 1-2.64-6.36M21 4v5h-5", size)

  def settings(size: Double = 15): StackPane =
    icon(
      size,
      svg("M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"),
      svg("M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z")
    )

  def sun(size: Double = 15): StackPane =
    icon(
      size,
      svg("M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z"),
      svg("M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41")
    )

  def moon(size: Double = 15): StackPane =
    path("M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z", size)

  def search(size: Double = 14): StackPane =
    path("M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16zM21 21l-4.35-4.35", size)

  def swap(size: Double = 15): StackPane =
    path("M7 4v16M7 4l-3 3M7 4l3 3M17 20V4M17 20l-3-3M17 20l3-3", size)

  def grid(size: Double = 15): StackPane =
    path("M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z", size)

  def dice(size: Double = 15): StackPane =
    icon(
      size,
      svg("M4 4h16v16H4z"),
      svg("M9 9h.01M15 9h.01M12 12h.01M9 15h.01M15 15h.01")
    )

  def recycle(size: Double = 15): StackPane =
    path("M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5", size)

  def copy(size: Double = 14): StackPane =
    path("M9 9h11v11H9zM5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1", size)

  def folder(size: Double = 14): StackPane =
    path("M3 6a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z", size)

  def trash(size: Double = 14): StackPane =
    path("M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14M10 11v5M14 11v5", size)

  def send(size: Double = 14): StackPane =
    path("M5 12h14M13 6l6 6-6 6", size)

  def restore(size: Double = 14): StackPane =
    path("M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5", size)

  def terminal(size: Double = 14): StackPane =
    path("M4 6h16v12H4zM8 10l3 2-3 2M13 14h3", size)

  def chevronDown(size: Double = 12): StackPane =
    path("M6 9l6 6 6-6", size)

  def chevronRight(size: Double = 12): StackPane =
    path("M9 6l6 6-6 6", size)

  def chevronLeft(size: Double = 18): StackPane =
    path("M15 6l-6 6 6 6", size)

  def maximize(size: Double = 14): StackPane =
    path("M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7", size)

  def sidebarToggle(size: Double = 15): StackPane =
    path("M4 4h16v16H4zM9 4v16", size)

  def sparkles(size: Double = 15): StackPane =
    path("M12 3l1.8 4.2L18 9l-4.2 1.8L12 15l-1.8-4.2L6 9l4.2-1.8zM19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9z", size)

  def image(size: Double = 15): StackPane =
    icon(
      size,
      svg("M3 5h18v14H3z"),
      svg("M8.5 10.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zM21 15l-5-5L5 19")
    )

  def brush(size: Double = 15): StackPane =
    path("M18 3l3 3-9.5 9.5-3.5.5.5-3.5zM6.5 17.5A2.5 2.5 0 0 0 4 20c0 1-1 1.5-1 1.5s2.5.5 4.5-.5a2.5 2.5 0 0 0-1-3.5z", size)

  def info(size: Double = 15): StackPane =
    path("M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 16v-4M12 8h.01", size)

  def wand(size: Double = 15): StackPane =
    path("M15 4l5 5L8 21l-5-5zM12 7l5 5", size)

  def columns(size: Double = 15): StackPane =
    path("M3 4h18v16H3zM12 4v16", size)

  def gallery(size: Double = 15): StackPane =
    path("M3 3h8v8H3zM13 3h8v8h-8zM3 13h8v8H3zM13 13h8v8h-8z", size)

  def globe(size: Double = 15): StackPane =
    path("M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM3.6 9h16.8M3.6 15h16.8M12 3a14 14 0 0 0 0 18M12 3a14 14 0 0 1 0 18", size)

  def palette(size: Double = 15): StackPane =
    path("M12 21a9 9 0 1 1 9-9c0 2-1.5 3-3 3h-1.5a1.5 1.5 0 0 0-1.5 1.5c0 .7.4 1.2.8 1.8.4.6.2 2.7-3.8 2.7zM7.5 10.5h.01M10.5 7.5h.01M15.5 8.5h.01", size)

  def sliders(size: Double = 15): StackPane =
    path("M4 7h10M18 7h2M4 17h4M12 17h8M14 4v6M8 14v6", size)

  def cpu(size: Double = 15): StackPane =
    path("M6 6h12v12H6zM9 9h6v6H9zM9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4", size)

  def lock(size: Double = 15): StackPane =
    path("M5 11h14v10H5zM8 11V7a4 4 0 0 1 8 0v4", size)

  def zap(size: Double = 14): StackPane =
    path("M13 2L4 14h7l-1 8 9-12h-7z", size)

  def layers(size: Double = 15): StackPane =
    path("M12 3l9 4.5-9 4.5-9-4.5zM3 12l9 4.5 9-4.5M3 16.5l9 4.5 9-4.5", size)

  def save(size: Double = 14): StackPane =
    path("M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2zM17 21v-8H7v8M7 3v5h8", size)

  def check(size: Double = 14): StackPane =
    path("M20 6L9 17l-5-5", size)

  def alert(size: Double = 14): StackPane =
    path("M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0zM12 9v4M12 17h.01", size)
