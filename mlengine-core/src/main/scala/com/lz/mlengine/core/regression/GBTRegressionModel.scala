package com.lz.mlengine.core.regression

import breeze.linalg.{DenseVector, Vector}
import com.lz.mlengine.core.{ModelLoader, RegressionModel}

class GBTRegressionModel(val trees: Array[DecisionTreeRegressionModel],
                         val weights: Array[Double],
                         override val featureToIndexMap: Map[String, Int]
                        ) extends RegressionModel(featureToIndexMap) {

  override private[mlengine] def predictImpl(vector: Vector[Double]): Vector[Double] = {
    val prediction = trees.zip(weights)
      .map(treeWithWeight =>
        treeWithWeight._1.predictImpl(vector)(0) * treeWithWeight._2)
      .sum
    DenseVector(prediction)
  }

}

object GBTRegressionModel extends ModelLoader[GBTRegressionModel]



