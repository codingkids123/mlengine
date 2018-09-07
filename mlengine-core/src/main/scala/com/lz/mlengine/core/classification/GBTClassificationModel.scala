package com.lz.mlengine.core.classification

import breeze.linalg.{DenseVector, Vector}
import com.lz.mlengine.core.{ClassificationModel, ModelLoader}
import com.lz.mlengine.core.regression.DecisionTreeRegressionModel

class GBTClassificationModel(val trees: Array[DecisionTreeRegressionModel], val weights: Array[Double],
                             override val featureToIndexMap: Map[String, Int],
                             override val indexToLabelMap: Map[Int, String]
                            ) extends ClassificationModel(featureToIndexMap, indexToLabelMap) {

  override private[mlengine] def predictImpl(vector: Vector[Double]): Vector[Double] = {
    val prediction = trees.zip(weights)
      .map(treeWithWeight =>
        treeWithWeight._1.predictImpl(vector)(0) * treeWithWeight._2)
      .sum
    val probability = 1.0 / (1.0 + math.exp(-2.0 * prediction))
    DenseVector[Double](1.0 - probability, probability)
  }

}

object GBTClassificationModel extends ModelLoader[GBTClassificationModel]
