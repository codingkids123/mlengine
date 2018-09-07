package com.lz.mlengine.core.classification

import breeze.linalg.{DenseVector, Vector}
import com.lz.mlengine.core.{ClassificationModel, ModelLoader}

class LinearSVCModel(val coefficients: Vector[Double], val intercept: Double,
                     override val featureToIndexMap: Map[String, Int],
                     override val indexToLabelMap: Map[Int, String]
                    ) extends ClassificationModel(featureToIndexMap, indexToLabelMap) {

  override private[mlengine] def predictImpl(vector: Vector[Double]): Vector[Double] = {
    val r = (coefficients dot vector) + intercept
    DenseVector(-r, r)
  }

}

object LinearSVCModel extends ModelLoader[LinearSVCModel]
