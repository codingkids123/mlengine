package com.lz.mlengine

import breeze.linalg._

class LogisticRegressionModel(val coefficients: Matrix[Double], val intercept: Vector[Double],
                              val featureToIndexMap: Map[String, Int], val indexToLabelMap: Map[Int, String]
                             ) extends MLModel(featureToIndexMap, Some(indexToLabelMap)) {

  override def toString = s"${coefficients} ${intercept}"

  def predictImpl(vector: Vector[Double]): Vector[Double] = {
    val r = coefficients * vector + intercept
    if (intercept.size == 1) {
      // Binary classification.
      new DenseVector(Array(1.0 / (1.0 + math.exp(r(0))), 1.0 / (1.0 + math.exp(-r(0)))))
    } else {
      // Multinomial classification.
      val maxValue = max(r)
      if (maxValue == Double.PositiveInfinity) {
          r.map(v => if (v == Double.PositiveInfinity) 1.0 else 0.0)
      } else {
        val exps =  if (maxValue > 0) {
          r.map(v => math.exp(v - maxValue))
        } else {
          r.map(v => math.exp(v))
        }
        exps / sum(exps)
      }
    }
  }

}

object LogisticRegressionModel extends MLModelLoader[LogisticRegressionModel]