package com.lz.mlengine.core

import java.io.{File, FileOutputStream}

import javax.imageio.ImageIO
import org.junit.Assert._
import org.junit.{Rule, Test}
import org.junit.rules.TemporaryFolder
import org.scalatest.junit.JUnitSuite
import org.scalatest.Matchers

class ReportTest extends JUnitSuite with Matchers {

  val _temporaryFolder = new TemporaryFolder

  @Rule
  def temporaryFolder = _temporaryFolder

  @Test def testPlotCurves() = {
    val file = new File(s"${temporaryFolder.getRoot.getPath}/plot.png")
    val fos = new FileOutputStream(file)
    val curves = Seq(
      ((0.0 to 1.0 by 0.1).map { x => (x, x * x)}, "y = x ^ 2", "x", "y"),
      ((0.0 to 1.0 by 0.1).map { x => (x, x * x / 2.0)}, "y = x ^ 2 / 2", "x", "y"),
      ((0.0 to 1.0 by 0.1).map { x => (x, x * x / 3.0)}, "y = x ^ 2 / 3", "x", "y")
    )
    val graph = Report.generateGraph(curves, "test figure")
    ImageIO.write(graph, "png", fos)
    fos.close()
    assertTrue(file.exists())
  }

  @Test def testGenerateClassificationDetailReport() = {
    val metrics = new ClassificationMetrics(
      Map(
        "p" -> Map(
          0.0 -> ConfusionMatrix(1, 1, 1, 0),
          0.5 -> ConfusionMatrix(0, 1, 1, 1)
        ),
        "n" -> Map(
          0.0 -> ConfusionMatrix(1, 1, 1, 0),
          0.5 -> ConfusionMatrix(0, 1, 1, 1)
        )
      )
    )
    val report = Report.generateDetail(metrics)
    val expected = Report(
      "label,threshold,accuracy,precision,recall,tpr,fpr,fscore",
      Seq("p,0.0000,0.6667,0.5000,1.0000,1.0000,0.5000,0.6667",
        "p,0.5000,0.3333,0.0000,0.0000,0.0000,0.5000,NaN",
        "n,0.0000,0.6667,0.5000,1.0000,1.0000,0.5000,0.6667",
        "n,0.5000,0.3333,0.0000,0.0000,0.0000,0.5000,NaN")
    )
    assertEquals(expected.toString(), report.toString())
  }

  @Test def testGenerateClassificationSummaryReport() = {
    val metrics = new ClassificationMetrics(
      Map(
        "p" -> Map(
          0.0 -> ConfusionMatrix(1, 1, 1, 0),
          0.5 -> ConfusionMatrix(0, 1, 1, 1)
        ),
        "n" -> Map(
          0.0 -> ConfusionMatrix(1, 1, 1, 0),
          0.5 -> ConfusionMatrix(0, 1, 1, 1)
        )
      )
    )
    val report = Report.generateSummary(metrics)
    val expected = Report(
      "label,precision@recall0.9,precision@recall0.8,precision@recall0.5," +
        "recall@precision0.9,recall@precision0.8,recall@precision0.5,auc",
      Seq("p,0.5000,0.5000,0.5000,NaN,NaN,1.0000,0.5000",
        "n,0.5000,0.5000,0.5000,NaN,NaN,1.0000,0.5000")
    )
    assertEquals(expected.toString(), report.toString())
  }

  @Test def testGenerateRegressionSummaryReport() = {
    val metrics = new RegressionMetrics(0.1, 0.2, 0.3, 0.4)
    val report = Report.generateSummary(metrics)
    val expected = Report(
      "explained variance,mean squared error,mean absolute error,r2",
      Seq("0.1000,0.2000,0.3000,0.4000")
    )
    assertEquals(expected.toString(), report.toString())
  }

}
