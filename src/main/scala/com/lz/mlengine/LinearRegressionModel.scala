package com.lz.mlengine

import breeze.linalg._

class LinearRegressionModel(val coefficients: Vector[Double], val intercept: Double, val scale: Double,
                            val featureToIndexMap: Map[String, Int]
                           ) extends MLModel(featureToIndexMap, None) {

  override def toString = s"${coefficients} ${intercept}"

  def predictImpl(vector: Vector[Double]): Vector[Double] = {
    DenseVector((coefficients dot vector) + intercept)
  }

}

object LinearRegressionModel extends MLModelLoader[LinearRegressionModel]
