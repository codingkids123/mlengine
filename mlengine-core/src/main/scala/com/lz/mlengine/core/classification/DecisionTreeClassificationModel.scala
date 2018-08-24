package com.lz.mlengine.core.classification

import breeze.linalg.{Vector, sum}
import com.lz.mlengine.core.tree.Node
import com.lz.mlengine.core.{MLModel, MLModelLoader}

class DecisionTreeClassificationModel(val rootNode: Node, val featureToIndexMap: Map[String, Int],
                                      val indexToLabelMap: Map[Int, String]
                                     ) extends MLModel(featureToIndexMap, Some(indexToLabelMap)) {

  override private[mlengine] def predictImpl(vector: Vector[Double]): Vector[Double] = {
    val prediction = rootNode.predictImpl(vector).impurityStats
    prediction / sum(prediction)
  }

}

object DecisionTreeClassificationModel extends MLModelLoader[DecisionTreeClassificationModel]
