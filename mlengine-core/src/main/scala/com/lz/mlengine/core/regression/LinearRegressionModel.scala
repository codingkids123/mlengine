package com.lz.mlengine.core.regression

import breeze.linalg._
import com.lz.mlengine.core.{ModelLoader, RegressionModel}

class LinearRegressionModel(val coefficients: Vector[Double], val intercept: Double, val scale: Double,
                            override val featureToIndexMap: Map[String, Int]
                           ) extends RegressionModel(featureToIndexMap) {

  override private[mlengine] def predictImpl(vector: Vector[Double]): Vector[Double] = {
    DenseVector((coefficients dot vector) + intercept)
  }

}

object LinearRegressionModel extends ModelLoader[LinearRegressionModel]
