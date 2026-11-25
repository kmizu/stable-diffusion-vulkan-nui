package sdfx.engine

import java.nio.file.{Files, Path}

/** Resolves worker dependencies from regular application jars and sbt's layered class loaders. */
object WorkerClasspath:
  def forResource(resource: String, anchorClass: Class[?]): Option[String] =
    val loader=Option(anchorClass.getClassLoader).getOrElse(ClassLoader.getSystemClassLoader)
    if loader.getResource(resource)==null then None
    else
      val paths=scala.collection.mutable.LinkedHashSet.empty[Path]
      def addUrl(url: java.net.URL): Unit =
        if url!=null && "file".equalsIgnoreCase(url.getProtocol) then
          try
            val p=Path.of(url.toURI).toAbsolutePath.normalize()
            if Files.exists(p) then paths+=p
          catch case _: Throwable => ()
      Option(anchorClass.getProtectionDomain).flatMap(pd => Option(pd.getCodeSource)).foreach(cs => addUrl(cs.getLocation))
      var current: ClassLoader=loader
      while current!=null do
        current match
          case ucl: java.net.URLClassLoader => ucl.getURLs.foreach(addUrl)
          case _ => ()
        current=current.getParent
      Option(System.getProperty("java.class.path")).getOrElse("").trim
        .split(java.util.regex.Pattern.quote(java.io.File.pathSeparator),-1).iterator
        .map(_.trim).filter(_.nonEmpty).foreach { raw =>
          try
            val p=Path.of(raw).toAbsolutePath.normalize()
            if Files.exists(p) then paths+=p
          catch case _: Throwable => ()
        }
      Option.when(paths.nonEmpty)(paths.map(_.toString).mkString(java.io.File.pathSeparator))
