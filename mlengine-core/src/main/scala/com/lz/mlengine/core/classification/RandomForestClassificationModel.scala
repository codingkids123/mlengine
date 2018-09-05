package com.lz.mlengine.core.classification

import breeze.linalg.{Vector, sum}
import com.lz.mlengine.core.{ClassificationModel, MLModelLoader}

class RandomForestClassificationModel(val trees: Array[DecisionTreeClassificationModel],
                                      val weights: Array[Double],
                                      override val featureToIndexMap: Map[String, Int],
                                      override val indexToLabelMap: Map[Int, String]
                                     ) extends ClassificationModel(featureToIndexMap, indexToLabelMap) {

  override private[mlengine] def predictImpl(vector: Vector[Double]): Vector[Double] = {
    val prediction = trees.zip(weights)
      .map(treeWithWeight =>
        treeWithWeight._1.predictImpl(vector) * treeWithWeight._2)
      .reduce((preciction1, preciction2) => preciction1 + preciction2)
    prediction / sum(prediction)
  }

}

object RandomForestClassificationModel extends MLModelLoader[RandomForestClassificationModel]