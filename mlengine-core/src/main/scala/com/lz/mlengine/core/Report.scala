package com.lz.mlengine.core

import java.awt.Graphics2D
import java.awt.image.BufferedImage
import java.nio.charset.Charset

import breeze.linalg._
import breeze.plot._

case class Report(header: String, report: Seq[String]) {

  override def toString(): String = {
    (Seq(header) ++ report).mkString("\n")
  }

}

object Report {

  type Curve = (Seq[(Double, Double)], String, String, String)

  val UTF_8 = Charset.forName("UTF-8")

  val PLOT_WIDTH = 600
  val PLOT_HEIGHT = 400
  val PLOT_COLS = 2

  def generateGraph(curves: Seq[Curve], title: String, dpi: Int = 72): BufferedImage = {
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

    // default dpi is 72
    val width = PLOT_WIDTH * cols
    val height =  PLOT_HEIGHT * rows
    val scale = dpi / 72.0
    val swidth = (width * scale).toInt
    val sheight = (height * scale).toInt

    val image = new BufferedImage(swidth,sheight,BufferedImage.TYPE_INT_ARGB)
    val g2d = image.createGraphics()
    g2d.scale(scale, scale)
    drawPlots(g2d)
    g2d.dispose
    image
  }

  def generatePrGrpah(metrics: ClassificationMetrics): BufferedImage = Report.generateGraph(
    metrics.labels.map { l => (metrics.prCurve(l), s"Label: $l", "precision", "recall") },
    "PR Curve"
  )

  def generateRocGrpah(metrics: ClassificationMetrics): BufferedImage = Report.generateGraph(
    metrics.labels.map { l => (metrics.rocCurve(l), s"Label: $l, AUC: ${metrics.areaUnderROC(l)}", "fpr", "tpr") },
    "ROC Curve"
  )

  def generateSummary(metrics: Metrics): Report = {
    metrics match {
      case m: ClassificationMetrics =>
        Report(
          "label,precision@recall0.9,precision@recall0.8,precision@recall0.5," +
            "recall@precision0.9,recall@precision0.8,recall@precision0.5,auc",
          m.confusionMatrices.keys.map { label =>
            s"$label,"+
            s"${format(m.precision(label, 0.9))},${format(m.precision(label, 0.8))}," +
              s"${format(m.precision(label, 0.5))},${format(m.recall(label, 0.9))}," +
              s"${format(m.recall(label, 0.8))},${format(m.recall(label, 0.5))}," +
              s"${format(m.areaUnderROC(label))}"
          }.toSeq
        )
      case m: RegressionMetrics =>
        Report(
          "explained variance,mean squared error,mean absolute error,r2",
          Seq(
            s"${format(m.explainedVariance)},${format(m.meanSquaredError)},${format(m.meanAbsoluteError)}," +
              s"${format(m.r2)}")
        )
    }
  }

  def generateDetail(metrics: ClassificationMetrics): Report = {
    Report(
      "label,threshold,accuracy,precision,recall,tpr,fpr,fscore",
      metrics.confusionMatrices.flatMap {
        case (label, matrices) =>
          matrices.map {
            case (threshold, matrix) =>
              s"$label," +
                s"${format(threshold)},${format(matrix.accuracy)},${format(matrix.precision)}," +
                s"${format(matrix.recall)},${format(matrix.tpr)},${format(matrix.fpr)},${format(matrix.fScore)}"
          }
      }.toSeq
    )
  }

  def mergeReports(reports: Seq[Report]): Report = {
    Report(reports.head.header, reports.flatMap(r => r.report))
  }

  def format(number: Any): String = {
    number match {
      case n: Double => f"$n%.4f"
      case n: Float => f"$n%.4f"
      case n: Int => f"$n%.4f"
      case Some(n: Double) => f"$n%.4f"
      case Some(n: Float) => f"$n%.4f"
      case Some(n: Int) => f"$n%.4f"
      case None => "NaN"
    }
  }
}
