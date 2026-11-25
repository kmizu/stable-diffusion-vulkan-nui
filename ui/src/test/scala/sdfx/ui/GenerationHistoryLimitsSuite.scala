package sdfx.ui

import munit.FunSuite
import sdfx.core.presets.{FieldKey, PresetAtom, PresetError, PresetRecord}
import sdfx.ui.presets.{GenerationHistory, GenerationHistoryStore}
import sdfx.ui.vm.Mode

import java.nio.file.{Files, Path}
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit
import scala.concurrent.duration.*

class GenerationHistoryLimitsSuite extends FunSuite:
  override val munitTimeout: Duration = 120.seconds

  private def runAsync[A](start: (A => Unit) => Unit): A =
    val future = new CompletableFuture[A]()
    start(a => future.complete(a))
    future.get(60, TimeUnit.SECONDS)

  test("GenerationHistoryStore saves 35 full 600-field snapshots exceeding legacy 100k node cap") {
    val dir = Files.createTempDirectory("gen-history-limit-test-")
    val store = new GenerationHistoryStore(dir, GenerationHistory.DefaultLimit, r => r())
    try
      val baseFields: Map[FieldKey, PresetAtom] = (0 until 600).map { j =>
        FieldKey(Vector("plugin", s"p$j", "value")) -> PresetAtom.Text(s"val-$j")
      }.toMap

      val count = 35 // > 126,000 JSON nodes (exceeds legacy 100,000 limit)
      for i <- 0 until count do
        val snapshot = baseFields.updated(FieldKey(Vector("generation", "seed")), PresetAtom.Integer(i.toLong))
        val res = runAsync[Either[PresetError, Option[PresetRecord]]](cb => store.record(Mode.Txt2Img, snapshot, cb))
        assert(res.isRight, s"record $i failed: $res")
        assert(res.toOption.flatten.isDefined, s"record $i unexpectedly deduplicated")

      val listed = runAsync[Either[PresetError, Vector[PresetRecord]]](cb => store.list(Mode.Txt2Img)(cb))
      assert(listed.isRight, s"reload failed: $listed")
      assertEquals(listed.toOption.get.size, count)
    finally
      store.close()
      val walk = Files.walk(dir)
      try walk.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists) finally walk.close()
  }
