package com.lz.mlengine.core.regression

import breeze.linalg.{DenseVector, Vector}
import com.lz.mlengine.core.tree.Node
import com.lz.mlengine.core.{MLModelLoader, RegressionModel}

class DecisionTreeRegressionModel(val rootNode: Node, override val featureToIndexMap: Map[String, Int]
                                 ) extends RegressionModel(featureToIndexMap) {

  override private[mlengine] def predictImpl(vector: Vector[Double]): Vector[Double] = {
    DenseVector(rootNode.predictImpl(vector).prediction)
  }

}

object DecisionTreeRegressionModel extends MLModelLoader[DecisionTreeRegressionModel]
