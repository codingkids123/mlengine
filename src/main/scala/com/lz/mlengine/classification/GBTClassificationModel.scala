package com.lz.mlengine.classification

import breeze.linalg.{DenseVector, Vector, sum}
import com.lz.mlengine.{MLModel, MLModelLoader}
import com.lz.mlengine.regression.DecisionTreeRegressionModel

class GBTClassificationModel(val trees: Array[DecisionTreeRegressionModel],
                             val weights: Array[Double],
                             val featureToIndexMap: Map[String, Int],
                             val indexToLabelMap: Map[Int, String]
                            ) extends MLModel(featureToIndexMap, Some(indexToLabelMap)) {

  override private[mlengine] def predictImpl(vector: Vector[Double]): Vector[Double] = {
    val prediction = trees.zip(weights)
      .map(treeWithWeight =>
        treeWithWeight._1.predictImpl(vector)(0) * treeWithWeight._2)
      .sum
    val probability = 1.0 / (1.0 + math.exp(-2.0 * prediction))
    DenseVector[Double](1.0 - probability, probability)
  }

}

object GBTClassificationModel extends MLModelLoader[GBTClassificationModel]
