package com.lz.mlengine.core

import java.awt.Graphics2D
import java.io.OutputStream
import java.nio.charset.Charset

import breeze.linalg._
import breeze.plot._

object Report {

  type Curve = (Seq[(Double, Double)], String, String, String)

  val UTF_8 = Charset.forName("UTF-8")

  val PLOT_WIDTH = 600
  val PLOT_HEIGHT = 400
  val PLOT_COLS = 2

  def plotCurves(curves: Seq[Curve], out: OutputStream, title: String, dpi: Int = 72) = {
    val cols = PLOT_COLS
    val rows = curves.length / cols + curves.length % cols

    val plots = curves.zipWithIndex.map {
      case ((curve, title, xLabel, yLabel), index) =>
        val p = new Plot()
        p += plot(DenseVector(curve.map(_._1).toArray), DenseVector(curve.map(_._2).toArray))
        p.title = title
        p.xlabel = xLabel
        p.ylabel = yLabel
        p
    }

    def drawPlots(g2d : Graphics2D) {
      var px = 0; var py = 0
      for (p <- plots) {
        p.chart.draw(g2d, new java.awt.Rectangle(px * PLOT_WIDTH, py * PLOT_HEIGHT, PLOT_WIDTH, PLOT_HEIGHT))
        px = (px +1) % cols
        if (px == 0) py = (py + 1) % rows
      }
    }

    ExportGraphics.writePNG(
      out,
      draw = drawPlots,
      width = PLOT_WIDTH * cols,
      height =  PLOT_HEIGHT * rows,
      dpi = dpi)
  }

  def generateReport(metrics: ClassificationMetrics, out: OutputStream) = {
    out.write("label,threshold,accuracy,precision,recall,tpr,fpr,fscore\n".getBytes(UTF_8))
    metrics.confusionMatrices.foreach {
      case (label, matrices) =>
        matrices.foreach {
          case (threshold, matrix) =>
            out.write(
              (s"$label," +
                f"$threshold%.4f,${matrix.accuracy}%.4f,${matrix.precision}%.4f,${matrix.recall}%.4f,${matrix.tpr}%.4f,${matrix.fpr}%.4f,${matrix.fScore}%.4f\n"
                ).getBytes(UTF_8))
        }
    }

  }

  def generateReport(metrics: RegressionMetrics, out: OutputStream) = {
    out.write("explained variance,mean squared error,mean absolute error,r2\n".getBytes(UTF_8))
    out.write(
      f"${metrics.explainedVariance}%.4f,${metrics.meanSquaredError}%.4f,${metrics.meanAbsoluteError}%.4f,${metrics.r2}%.4f\n"
        .getBytes(UTF_8)
    )
  }

}
