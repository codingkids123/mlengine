package com.lz.mlengine.core

import java.io.{File, FileOutputStream}

import org.junit.Assert._
import org.junit.{Rule, Test}
import org.junit.rules.TemporaryFolder
import org.scalatest.junit.JUnitSuite
import org.scalatest.Matchers

import scala.io.Source

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
    Report.plotCurves(curves, fos, "test figure")
    fos.close()
    assertTrue(file.exists())
  }

  @Test def testGenerateClassificationReport() = {
    val file = new File(s"${temporaryFolder.getRoot.getPath}/classification.csv")
    val fos = new FileOutputStream(file)
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
    Report.generateReport(metrics, fos)
    fos.close()
    val fileContents = Source.fromFile(file.getAbsolutePath).getLines.mkString("\n")
    val expected =
      "label,threshold,accuracy,precision,recall,tpr,fpr,fscore\n" +
        "p,0.0000,0.6667,0.5000,1.0000,1.0000,0.5000,0.6667\n" +
        "p,0.5000,0.3333,0.0000,0.0000,0.0000,0.5000,NaN\n" +
        "n,0.0000,0.6667,0.5000,1.0000,1.0000,0.5000,0.6667\n" +
        "n,0.5000,0.3333,0.0000,0.0000,0.0000,0.5000,NaN"
    assertEquals(expected, fileContents)
  }

  @Test def testGenerateRegressionReport() = {
    val file = new File(s"${temporaryFolder.getRoot.getPath}/regression.csv")
    val fos = new FileOutputStream(file)
    val metrics = new RegressionMetrics(0.1, 0.2, 0.3, 0.4)
    Report.generateReport(metrics, fos)
    fos.close()
    val fileContents = Source.fromFile(file.getAbsolutePath).getLines.mkString("\n")
    val expected =
      "explained variance,mean squared error,mean absolute error,r2\n" +
        "0.1000,0.2000,0.3000,0.4000"
    assertEquals(expected, fileContents)
  }

}
