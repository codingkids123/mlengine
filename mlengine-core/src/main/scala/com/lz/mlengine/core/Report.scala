package com.lz.mlengine.core

import java.io.OutputStream
import java.nio.charset.Charset

import breeze.linalg._
import breeze.plot._

object Report {

  type Curve = (Seq[(Double, Double)], String, String, String)

  val UTF_8 = Charset.forName("UTF-8")

  def plotCurves(curves: Seq[Curve], out: OutputStream, title: String, dpi: Int = 72) = {
    val f = Figure()

    f.height = 400 * curves.length
    curves.zipWithIndex.map {
      case ((curve, title, xLabel, yLabel), index) =>
        val p = f.subplot(curves.length, 1, index)
        p += plot(DenseVector(curve.map(_._1).toArray), DenseVector(curve.map(_._2).toArray))
        p.title = title
        p.xlabel = xLabel
        p.ylabel = yLabel
    }

    f.refresh()
    ExportGraphics.writePNG(
      out,
      draw = f.drawPlots,
      width = f.width,
      height = f.height,
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
