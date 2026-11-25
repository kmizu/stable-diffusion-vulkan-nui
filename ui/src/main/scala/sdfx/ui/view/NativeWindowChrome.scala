package sdfx.ui.view

import javafx.stage.Stage

import java.awt.GraphicsEnvironment
import java.lang.reflect.Proxy
import java.nio.charset.StandardCharsets
import scala.util.control.NonFatal

/** Sets Windows 10/11 DWM immersive dark mode, caption color, and border colors on native Stage title bars.
  * Implemented with reflection over `java.lang.foreign` (JDK 22+) so the project still compiles cleanly on JDK 17/21.
  */
object NativeWindowChrome:
  private val DwmwaUseImmersiveDarkMode = 20
  private val DwmwaBorderColor = 34
  private val DwmwaCaptionColor = 35
  private val DwmwaTextColor = 36

  // BGR COLORREF values matching dark.css / light.css -sdfx-chrome, -sdfx-text, and subtle border tokens.
  private val DarkCaptionBgr = 0x0024201E
  private val DarkTextBgr = 0x00F2EEEC
  private val DarkBorderBgr = 0x0038312E
  private val LightCaptionBgr = 0x00FBF9F8
  private val LightTextBgr = 0x001F1A18
  private val LightBorderBgr = 0x00E2DAD6

  private val isWindows: Boolean =
    System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT).contains("win")

  private val supportsFfm: Boolean =
    isWindows && Runtime.version().feature() >= 22 && !GraphicsEnvironment.isHeadless

  private final class Handles(
      val arenaClass: Class[?],
      val memorySegmentClass: Class[?],
      val valueLayoutClass: Class[?],
      val addressLayout: AnyRef,
      val intLayout: AnyRef,
      val findWindowExW: java.lang.invoke.MethodHandle,
      val getWindowThreadProcessId: java.lang.invoke.MethodHandle,
      val dwmSetWindowAttribute: java.lang.invoke.MethodHandle
  )

  private lazy val handles: Option[Handles] =
    if !supportsFfm then None
    else
      try
        val arenaClass = Class.forName("java.lang.foreign.Arena")
        val linkerClass = Class.forName("java.lang.foreign.Linker")
        val lookupClass = Class.forName("java.lang.foreign.SymbolLookup")
        val fdClass = Class.forName("java.lang.foreign.FunctionDescriptor")
        val mlClass = Class.forName("java.lang.foreign.MemoryLayout")
        val msClass = Class.forName("java.lang.foreign.MemorySegment")
        val vlClass = Class.forName("java.lang.foreign.ValueLayout")
        val addrLayout = vlClass.getField("ADDRESS").get(null)
        val intLayout = vlClass.getField("JAVA_INT").get(null)
        val linker = linkerClass.getMethod("nativeLinker").invoke(null)
        val globalArena = arenaClass.getMethod("global").invoke(null)
        val libLookup = lookupClass.getMethod("libraryLookup", classOf[String], arenaClass)
        val user32 = libLookup.invoke(null, "user32", globalArena)
        val dwmapi = libLookup.invoke(null, "dwmapi", globalArena)

        val layoutArray = java.lang.reflect.Array.newInstance(mlClass, 4)
        java.lang.reflect.Array.set(layoutArray, 0, addrLayout)
        java.lang.reflect.Array.set(layoutArray, 1, addrLayout)
        java.lang.reflect.Array.set(layoutArray, 2, addrLayout)
        java.lang.reflect.Array.set(layoutArray, 3, addrLayout)
        val fdFind = fdClass.getMethod("of", mlClass, layoutArray.getClass).invoke(null, addrLayout, layoutArray)

        val pidLayouts = java.lang.reflect.Array.newInstance(mlClass, 2)
        java.lang.reflect.Array.set(pidLayouts, 0, addrLayout)
        java.lang.reflect.Array.set(pidLayouts, 1, addrLayout)
        val fdPid = fdClass.getMethod("of", mlClass, pidLayouts.getClass).invoke(null, intLayout, pidLayouts)

        val dwmLayouts = java.lang.reflect.Array.newInstance(mlClass, 4)
        java.lang.reflect.Array.set(dwmLayouts, 0, addrLayout)
        java.lang.reflect.Array.set(dwmLayouts, 1, intLayout)
        java.lang.reflect.Array.set(dwmLayouts, 2, addrLayout)
        java.lang.reflect.Array.set(dwmLayouts, 3, intLayout)
        val fdDwm = fdClass.getMethod("of", mlClass, dwmLayouts.getClass).invoke(null, intLayout, dwmLayouts)

        val findFn = lookupClass.getMethod("find", classOf[String])
        val downcall = linkerClass.getMethod("downcallHandle", msClass, fdClass, java.lang.reflect.Array.newInstance(Class.forName("java.lang.foreign.Linker$Option"), 0).getClass)
        val emptyOpts = java.lang.reflect.Array.newInstance(Class.forName("java.lang.foreign.Linker$Option"), 0)

        def bind(lib: AnyRef, name: String, fd: AnyRef): java.lang.invoke.MethodHandle =
          val opt = findFn.invoke(lib, name).asInstanceOf[java.util.Optional[AnyRef]]
          val sym = opt.orElseThrow(() => new UnsatisfiedLinkError(name))
          downcall.invoke(linker, sym, fd, emptyOpts).asInstanceOf[java.lang.invoke.MethodHandle]

        Some(
          new Handles(
            arenaClass,
            msClass,
            vlClass,
            addrLayout,
            intLayout,
            bind(user32, "FindWindowExW", fdFind),
            bind(user32, "GetWindowThreadProcessId", fdPid),
            bind(dwmapi, "DwmSetWindowAttribute", fdDwm)
          )
        )
      catch case NonFatal(_) => None

  /** Applies dark or light native window chrome to `stage` if running on Windows with JDK 22+. Safe no-op elsewhere. */
  def apply(stage: Stage, theme: String): Unit =
    if stage == null || !stage.isShowing then return
    val title = Option(stage.getTitle).getOrElse("")
    if title.isEmpty then return
    handles.foreach { h =>
      try
        val arena = h.arenaClass.getMethod("ofConfined").invoke(null).asInstanceOf[AutoCloseable]
        try
          val nullSeg = h.memorySegmentClass.getField("NULL").get(null)
          val utf16Bytes = (title + "\u0000").getBytes(StandardCharsets.UTF_16LE)
          val byteLayout = h.valueLayoutClass.getField("JAVA_BYTE").get(null)
          val titleSeg = h.arenaClass.getMethod("allocate", classOf[Long], classOf[Long])
            .invoke(arena, Long.box(utf16Bytes.length.toLong), Long.box(2L))
          val copyFrom = h.memorySegmentClass.getMethod("copyFrom", h.memorySegmentClass)
          val ofArray = h.memorySegmentClass.getMethod("ofArray", classOf[Array[Byte]])
          copyFrom.invoke(titleSeg, ofArray.invoke(null, utf16Bytes))

          val pidSeg = h.arenaClass.getMethod("allocate", Class.forName("java.lang.foreign.MemoryLayout"))
            .invoke(arena, h.intLayout)
          val currentPid = ProcessHandle.current().pid().toInt
          val getInt = h.memorySegmentClass.getMethod("get", Class.forName("java.lang.foreign.ValueLayout$OfInt"), classOf[Long])
          val setInt = h.memorySegmentClass.getMethod("set", Class.forName("java.lang.foreign.ValueLayout$OfInt"), classOf[Long], classOf[Int])
          val addressMethod = h.memorySegmentClass.getMethod("address")

          var prev: Any = nullSeg
          var matched: Any = null
          var guard = 0
          while matched == null && guard < 32 do
            guard += 1
            val found = h.findWindowExW.invoke(nullSeg, prev, nullSeg, titleSeg)
            val addr = addressMethod.invoke(found).asInstanceOf[Long]
            if addr == 0L then guard = 32
            else
              h.getWindowThreadProcessId.invoke(found, pidSeg)
              val ownerPid = getInt.invoke(pidSeg, h.intLayout, Long.box(0L)).asInstanceOf[Int]
              if ownerPid == currentPid then matched = found
              else prev = found

          if matched != null then
            val valSeg = h.arenaClass.getMethod("allocate", Class.forName("java.lang.foreign.MemoryLayout"))
              .invoke(arena, h.intLayout)
            def setAttr(attr: Int, value: Int): Unit =
              setInt.invoke(valSeg, h.intLayout, Long.box(0L), Int.box(value))
              h.dwmSetWindowAttribute.invoke(matched, Int.box(attr), valSeg, Int.box(4))
              ()

            val isLight = theme == "light"
            setAttr(DwmwaUseImmersiveDarkMode, if isLight then 0 else 1)
            setAttr(DwmwaCaptionColor, if isLight then LightCaptionBgr else DarkCaptionBgr)
            setAttr(DwmwaTextColor, if isLight then LightTextBgr else DarkTextBgr)
            setAttr(DwmwaBorderColor, if isLight then LightBorderBgr else DarkBorderBgr)
        finally arena.close()
      catch case NonFatal(_) => ()
    }
