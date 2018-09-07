package com.lz.mlengine.core.classification

import breeze.linalg.{Vector, sum}
import com.lz.mlengine.core.tree.Node
import com.lz.mlengine.core.{ClassificationModel, ModelLoader}

class DecisionTreeClassificationModel(val rootNode: Node, override val featureToIndexMap: Map[String, Int],
                                      override val indexToLabelMap: Map[Int, String]
                                     ) extends ClassificationModel(featureToIndexMap, indexToLabelMap) {

  override private[mlengine] def predictImpl(vector: Vector[Double]): Vector[Double] = {
    val prediction = rootNode.predictImpl(vector).impurityStats
    prediction / sum(prediction)
  }

}

object DecisionTreeClassificationModel extends ModelLoader[DecisionTreeClassificationModel]
