package com.lz.mlengine.spark

import breeze.linalg.{CSCMatrix, DenseMatrix, DenseVector, Matrix, SparseVector, Vector, VectorBuilder}
import org.apache.spark.ml.TreeConverter
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel => SparkDecisionTreeClassificationModel}
import org.apache.spark.ml.classification.{GBTClassificationModel => SparkGBTClassificationModel}
import org.apache.spark.ml.classification.{LinearSVCModel => SparkLinearSVCModel, LogisticRegressionModel => SparkLogisticRegressionModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel => SparkRandomForestClassificationModel}
import org.apache.spark.ml.linalg.{DenseMatrix => SparkDenseMatrix}
import org.apache.spark.ml.linalg.{DenseVector => SparkDenseVector}
import org.apache.spark.ml.linalg.Matrices
import org.apache.spark.ml.linalg.{Matrix => SparkMatrix}
import org.apache.spark.ml.linalg.{SparseMatrix => SparkSparseMatrix}
import org.apache.spark.ml.linalg.{SparseVector => SparkSparseVector}
import org.apache.spark.ml.linalg.{Vector => SparkVector}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel => SparkDecisionTreeRegressionModel}
import org.apache.spark.ml.regression.{GBTRegressionModel => SparkGBTRegressionModel}
import org.apache.spark.ml.regression.{LinearRegressionModel => SparkLinearRegressionModel}
import org.apache.spark.ml.regression.{RandomForestRegressionModel => SparkRandomForestRegressionModel}

import com.lz.mlengine.core.classification._
import com.lz.mlengine.core.regression._

object Converter {

  implicit def convert(matrix: SparkMatrix): Matrix[Double] = {
    matrix match {
      case m: SparkDenseMatrix => new DenseMatrix(m.numRows, m.numCols, m.toArray)
      case m: SparkSparseMatrix =>
        val builder = new CSCMatrix.Builder[Double](rows=m.numRows, cols=m.numCols)
        m.foreachActive((row, col, value) => builder.add(row, col, value))
        builder.result
    }
  }

  implicit def convert(matrix: Matrix[Double]): SparkMatrix = {
    matrix match {
      case m: DenseMatrix[Double] => Matrices.dense(m.rows, m.cols, m.toArray)
      case m: CSCMatrix[Double] => Matrices.sparse(m.rows, m.cols, m.colPtrs, m.rowIndices, m.data)
    }
  }

  implicit def convert(vector: SparkVector): Vector[Double] = {
    vector match {
      case v: SparkDenseVector => new DenseVector[Double](v.toArray)
      case v: SparkSparseVector =>
        val builder = new VectorBuilder[Double](v.size)
        v.foreachActive((idx, value) => builder.add(idx, value))
        builder.toSparseVector
    }
  }

  implicit def convert(vector: Vector[Double]): SparkVector = {
    vector match {
      case v: DenseVector[Double] =>  Vectors.dense(v.toArray)
      case v: SparseVector[Double] => Vectors.sparse(v.array.size, v.array.index, v.array.data)
    }
  }

  implicit def convert(model: SparkDecisionTreeClassificationModel)
                      (implicit featureToIndexMap: Map[String, Int], indexToLabelMap: Map[Int, String]
                      ): DecisionTreeClassificationModel = {
    val rootNode = TreeConverter.convertDecisionTree(model.rootNode)
    new DecisionTreeClassificationModel(rootNode, featureToIndexMap, indexToLabelMap)
  }

  implicit def convert(model: SparkGBTClassificationModel)
                      (implicit featureToIndexMap: Map[String, Int], indexToLabelMap: Map[Int, String]
                      ): GBTClassificationModel = {
    val trees = model.trees
      .map(tree =>
        new DecisionTreeRegressionModel(
          TreeConverter.convertDecisionTree(tree.rootNode),
          Map[String, Int]())
      )
    new GBTClassificationModel(trees, model.treeWeights, featureToIndexMap, indexToLabelMap)
  }

  implicit def convert(model: SparkLogisticRegressionModel)
                      (implicit featureToIndexMap: Map[String, Int], indexToLabelMap: Map[Int, String]
                      ): LogisticRegressionModel = {
    new LogisticRegressionModel(model.coefficientMatrix, model.interceptVector, featureToIndexMap, indexToLabelMap)
  }

  implicit def convert(model: SparkLinearSVCModel)
                      (implicit featureToIndexMap: Map[String, Int], indexToLabelMap: Map[Int, String]
                      ): LinearSVCModel = {
    new LinearSVCModel(model.coefficients, model.intercept, featureToIndexMap, indexToLabelMap)
  }

  implicit def convert(model: SparkRandomForestClassificationModel)
                      (implicit featureToIndexMap: Map[String, Int], indexToLabelMap: Map[Int, String]
                      ): RandomForestClassificationModel = {
    val trees = model.trees
      .map(tree =>
        new DecisionTreeClassificationModel(
          TreeConverter.convertDecisionTree(tree.rootNode),
          Map[String, Int](),
          Map[Int, String]())
      )
    new RandomForestClassificationModel(trees, model.treeWeights, featureToIndexMap, indexToLabelMap)
  }

  implicit def convert(model: SparkDecisionTreeRegressionModel)
                      (implicit featureToIndexMap: Map[String, Int]): DecisionTreeRegressionModel = {
    val rootNode = TreeConverter.convertDecisionTree(model.rootNode)
    new DecisionTreeRegressionModel(rootNode, featureToIndexMap)
  }

  implicit def convert(model: SparkGBTRegressionModel)
                      (implicit featureToIndexMap: Map[String, Int]): GBTRegressionModel = {
    val trees = model.trees
      .map(tree =>
        new DecisionTreeRegressionModel(
          TreeConverter.convertDecisionTree(tree.rootNode),
          Map[String, Int]())
      )
    new GBTRegressionModel(trees, model.treeWeights, featureToIndexMap)
  }


  implicit def convert(model: SparkLinearRegressionModel)
                      (implicit featureToIndexMap: Map[String, Int]): LinearRegressionModel = {
    new LinearRegressionModel(model.coefficients, model.intercept, model.scale, featureToIndexMap)
  }

  implicit def convert(model: SparkRandomForestRegressionModel)
                      (implicit featureToIndexMap: Map[String, Int]): RandomForestRegressionModel = {
    val trees = model.trees
      .map(tree =>
        new DecisionTreeRegressionModel(
          TreeConverter.convertDecisionTree(tree.rootNode),
          Map[String, Int]()
        )
      )
    new RandomForestRegressionModel(trees, model.treeWeights, featureToIndexMap)
  }

}
