package sdfx.scalatorch

import java.io.{BufferedReader, InputStreamReader, OutputStream}
import java.nio.charset.StandardCharsets.UTF_8
import sdfx.engine.JvmDryRunWorker

/** Compatibility entry point for existing scala-torch dry-run launches. */
object NativeDryRunWorker:
  private val version = "scala-torch-0.1.3-dryrun"
  val Features: Seq[String] = JvmDryRunWorker.Features
  def helloObject: ujson.Obj = JvmDryRunWorker.helloObject(version)
  def main(args: Array[String]): Unit =
    serve(new BufferedReader(new InputStreamReader(System.in, UTF_8)), System.out)
  def serve(in: BufferedReader, out: OutputStream): Unit = JvmDryRunWorker.serve(in, out, version)
