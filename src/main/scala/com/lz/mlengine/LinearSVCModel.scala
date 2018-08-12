package com.lz.mlengine

import breeze.linalg.{DenseVector, Vector}

class LinearSVCModel(val coefficients: Vector[Double], val intercept: Double,
                     val featureToIndexMap: Map[String, Int], val indexToLabelMap: Map[Int, String]
                    ) extends MLModel(featureToIndexMap, Some(indexToLabelMap)) {

  override private[mlengine] def predictImpl(vector: Vector[Double]): Vector[Double] = {
    val r = (coefficients dot vector) + intercept
    DenseVector(-r, r)
  }

}

object LinearSVCModel extends MLModelLoader[LinearSVCModel]
