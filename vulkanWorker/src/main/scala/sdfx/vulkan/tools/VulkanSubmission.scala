// Adapted from kmizu/scala-torch (BSD-3-Clause). See META-INF/LICENSE-scala-torch.txt.
package sdfx.vulkan.tools

import scala.util.Using
import scala.util.control.NonFatal
import NativeJson.*

private[tools] object VulkanSubmission:
  private val DestinationHost = "logs3.papertrailapp.com"
  private val DestinationPort = 32554

  private def compactValue(v: NativeJson): String = v match
    case Obj(fields) => fields.map { (k, item) => s"$k:${compactValue(item)}" }.mkString(" ")
    case Arr(items)  => items.map(compactValue).mkString(",")
    case Str(s)      => s
    case Number(n)   => n.toString
    case Bool(b)     => b.toString
    case Null        => ""

  private def sanitizeField(s: String): String =
    s.replace('|', '/').replace('\r', ' ').replace('\n', ' ').replace('\u0000', ' ')

  def submissionPreview(p: Map[String, NativeJson]): Map[String, NativeJson] =
    val username = p.get("username") match
      case Some(Str(s)) if s.trim.nonEmpty && s.length <= 128 && !s.exists(c => c == '\r' || c == '\n' || c == '|' || c == '\u0000') =>
        s.trim
      case _ =>
        throw IllegalArgumentException("Enter a username without line breaks or separators to preview submission")
    val record = p.get("record") match
      case Some(Obj(fields)) => fields
      case _                 => throw IllegalArgumentException("Choose a benchmark record to submit")
    val results = record.get("results") match
      case Some(Arr(items)) if items.forall(_.isInstanceOf[Obj]) => items.map(_.asInstanceOf[Obj].fields)
      case Some(Arr(_))                                          => throw IllegalArgumentException("Malformed benchmark record")
      case _                                                     => throw IllegalArgumentException("Choose a benchmark record to submit")
    val system = record.getOrElse("system", Obj(Map.empty)) match
      case Obj(fields) => fields
      case _           => throw IllegalArgumentException("Malformed benchmark record")
    val config = record.getOrElse("config", Obj(Map.empty)) match
      case Obj(fields) => fields
      case _           => throw IllegalArgumentException("Malformed benchmark record")
    val fingerprint = record.getOrElse("fingerprint", Str("")) match
      case Str(s) => s
      case _      => throw IllegalArgumentException("Malformed benchmark record")
    val includeHostname = p.getOrElse("include_hostname", Bool(false)) match
      case Bool(b) => b
      case _       => throw IllegalArgumentException("include_hostname must be boolean")
    val hostname =
      if includeHostname then
        try java.net.InetAddress.getLocalHost.getHostName
        catch case NonFatal(_) => "anonymous"
      else "anonymous"
    val runtime = system.get("runtime").collect { case Obj(f) => f }.getOrElse(Map.empty)
    val model = system.get("model").collect { case Obj(f) => f }.getOrElse(Map.empty)
    def strField(m: Map[String, NativeJson], k: String, default: String = ""): String =
      m.get(k).map(compactValue).getOrElse(default)
    val optimizations = Vector(
      "pipeline" -> strField(runtime, "pipeline"),
      "crossattention" -> strField(runtime, "attention"),
      "flags" -> compactValue(runtime.getOrElse("flags", Arr(Vector.empty))),
      "compute" -> strField(runtime, "backend"),
      "compile" -> "none",
      "dtype" -> strField(runtime, "dtype")
    ).map((k, v) => s"$k:$v").mkString(" ")
    val settings = Vector(
      "pipeline" -> strField(runtime, "pipeline"),
      "model" -> strField(model, "name"),
      "vae" -> strField(model, "vae_name"),
      "width" -> strField(config, "width", "512"),
      "height" -> strField(config, "height", "512")
    ).map((k, v) => s"$k:$v").mkString(" ")
    val ips = results.map(r => r.get("iterations_per_second").map(compactValue).getOrElse("error")).mkString(" / ")
    val rawFields = Vector(
      strField(record, "timestamp"),
      ips,
      compactValue(system.getOrElse("version", Obj(Map.empty))),
      compactValue(system.getOrElse("platform", Obj(Map.empty))),
      compactValue(system.getOrElse("libraries", Obj(Map.empty))),
      compactValue(system.getOrElse("gpu", Obj(Map.empty))),
      optimizations,
      settings,
      username,
      strField(record, "note"),
      fingerprint.take(6)
    )
    val baseLine = rawFields.map(sanitizeField).mkString("|").replace("  ", " ").replace("\"", "").trim
    val md = java.security.MessageDigest.getInstance("SHA-256")
    val lineHash = java.util.HexFormat.of().formatHex(md.digest(baseLine.getBytes(java.nio.charset.StandardCharsets.UTF_8))).take(6)
    val line = s"$baseLine|$lineHash"
    val stamp = java.time.ZonedDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("MMM dd HH:mm:ss", java.util.Locale.US))
    val packet = s"<14>$stamp $hostname SDBENCHMARK: $username $line\u0000"
    require(
      packet.getBytes(java.nio.charset.StandardCharsets.UTF_8).length <= 60000,
      "Benchmark submission exceeds the UDP payload limit"
    )
    val tokenBytes = new Array[Byte](24)
    new java.security.SecureRandom().nextBytes(tokenBytes)
    val token = java.util.Base64.getUrlEncoder.withoutPadding().encodeToString(tokenBytes)
    Map(
      "token" -> Str(token),
      "destination" -> Str(s"$DestinationHost:$DestinationPort/udp"),
      "packets" -> Arr(Vector(Str(packet)))
    )

  def submit(
      p: Map[String, NativeJson],
      pending: Option[(Long, Map[String, NativeJson])],
      sendUdp: (Vector[String], String, Int) => Unit = defaultSendUdp
  ): Map[String, NativeJson] =
    val valid = pending.filter { case (startedAt, preview) =>
      val elapsedS = (System.nanoTime() - startedAt) / 1e9
      elapsedS <= 300.0 && p.get("token").contains(preview("token"))
    }
    val (_, preview) = valid.getOrElse(
      throw IllegalArgumentException("Preview the exact benchmark payload again before submitting")
    )
    val packets = preview("packets").asInstanceOf[Arr].values.map(_.asInstanceOf[Str].value)
    try sendUdp(packets, DestinationHost, DestinationPort)
    catch
      case NonFatal(error) =>
        throw new NativeBackendError(
          "unsupported",
          s"Legacy submission unavailable: ${error.getMessage}; export the benchmark JSON locally"
        )
    Map(
      "status" -> Str("sent_unconfirmed"),
      "message" -> Str("UDP packet sent. The legacy logger provides no receipt confirmation.")
    )

  private def defaultSendUdp(packets: Vector[String], host: String, port: Int): Unit =
    Using.resource(new java.net.DatagramSocket()) { socket =>
      socket.setSoTimeout(5000)
      val addr = java.net.InetAddress.getByName(host)
      packets.foreach { pkt =>
        val bytes = pkt.getBytes(java.nio.charset.StandardCharsets.UTF_8)
        val datagram = new java.net.DatagramPacket(bytes, bytes.length, addr, port)
        socket.send(datagram)
      }
    }
