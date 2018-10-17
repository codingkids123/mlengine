package com.lz.mlengine.core

case class ConfusionMatrix(tp: Int, fp: Int, tn: Int, fn: Int) {

  require(tp >= 0 && fp >= 0 && tn >= 0 && fn >= 0)

  def precision = tp.toDouble / (tp + fp)

  def recall = tp.toDouble / (tp + fn)

  def tpr = recall

  def fpr = fp.toDouble / (fp + tn)

  def accuracy = (tp + tn).toDouble / (tp + fp + tn + fn)

  def fScore = 2 * precision * recall / (precision + recall)

}

abstract class Metrics

class ClassificationMetrics(val confusionMatrices: Map[String, Map[Double, ConfusionMatrix]]
                           ) extends Metrics with Serializable {

  def labels = confusionMatrices.keys.toSeq

  def prCurve(label: String): Seq[(Double, Double)] =
    confusionMatrices.get(label).get.values.toSeq.map(m => (m.recall, m.precision)).sortBy(_._1)

  def rocCurve(label: String): Seq[(Double, Double)] =
    Seq((0.0, 0.0)) ++ confusionMatrices.get(label).get.values.toSeq.map(m => (m.fpr, m.tpr)).sortBy(_._1) ++
      Seq((1.0, 1.0))

  def areaUnderROC(label: String): Double =
    rocCurve(label).sliding(2).map { case Seq(roc1, roc2) => (roc2._1 - roc1._1) * (roc1._2 + roc2._2) / 2 }.sum

  def precision(label: String, recall: Double): Option[Double] = {
    prCurve(label).find(_._1 >= recall).map(_._2)
  }

  def recall(label: String, precision: Double): Option[Double] = {
    prCurve(label).sortBy(_._2).find(_._2 >= precision).map(_._1)
  }

}

class RegressionMetrics(val explainedVariance: Double, val meanSquaredError: Double, val meanAbsoluteError: Double,
                        val r2: Double) extends Metrics with Serializable