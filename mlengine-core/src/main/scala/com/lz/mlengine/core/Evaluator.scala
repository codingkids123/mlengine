package com.lz.mlengine.core

object Evaluator {

  def evaluate(predictions: Seq[(Map[String, Double], String)], labels: Seq[String], numSteps: Int
              ): ClassificationMetrics = {
    val confusionMatrices = labels.map { label =>
      val scores = predictions.map { case (prediction, _) => prediction.get(label).get }
      val maxScore = scores.max
      val minScore = scores.min
      val step = (maxScore - minScore) / numSteps

      val thresholdToConfusionMatrixMap = (minScore + step until maxScore by step).map {
        threshold =>
          var tp = 0
          var fp = 0
          var tn = 0
          var fn = 0
          predictions.foreach {
            case (predictions, truth) =>
              val score = predictions.get(label).get
              if (score > threshold && label == truth) {
                tp += 1
              } else if (score > threshold && label != truth) {
                fp += 1
              } else if (score <= threshold && label != truth) {
                tn += 1
              } else {
                fn += 1
              }
          }
          (threshold, ConfusionMatrix(tp, fp, tn, fn))
      }.toMap
      (label, thresholdToConfusionMatrixMap)
    }.toMap
    new ClassificationMetrics(confusionMatrices)
  }

  def evaluate(predictions: Seq[(Double, Double)]): RegressionMetrics = {
    val numSamples = predictions.length
    val mean = predictions.map { case (_, truth) => truth }.sum / numSamples
    val variance = predictions.map { case (_, truth) => math.pow(truth - mean, 2) }.sum / numSamples
    val explainedVariance = predictions.map { case (prediction, _) => prediction - mean}.sum / numSamples
    val mse = predictions.map { case (prediction, truth) => math.pow((prediction - truth), 2) }.sum / numSamples
    val mae = predictions.map { case (prediction, truth) => math.abs(prediction - truth) }.sum / numSamples
    val r2 = 1 - mse / variance
    new RegressionMetrics(explainedVariance, mse, mae, r2)
  }

}
