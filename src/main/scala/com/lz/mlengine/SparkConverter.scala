package com.lz.mlengine

import breeze.linalg.{DenseMatrix, DenseVector, Matrix, Vector}
import org.apache.spark.ml.classification.{LogisticRegressionModel => SparkLogisticRegressionModel}
import org.apache.spark.ml.linalg.{Matrix => SparkMatrix}
import org.apache.spark.ml.linalg.{Vector => SparkVector}
import org.apache.spark.ml.linalg.Vectors

object SparkConverter {

  implicit def convert(matrix: SparkMatrix): Matrix[Double] = {
    new DenseMatrix(matrix.numRows, matrix.numCols, matrix.toArray)
  }

  implicit def convert(vector: SparkVector): Vector[Double] = {
    new DenseVector[Double](vector.toArray)
  }

  implicit def convert(vector: Vector[Double]): SparkVector = {
    Vectors.dense(vector.toArray)
  }

  implicit def convert(model: SparkLogisticRegressionModel)
                      (implicit featureToIndexMap: Map[String, Int], indexToLabelMap: Map[Int, String]): MLModel = {
    new LogisticRegressionModel(
      model.coefficientMatrix, model.interceptVector, featureToIndexMap, indexToLabelMap
    )
  }

}
