package org.apache.spark.ml

import breeze.linalg.DenseVector
import breeze.linalg.Vector
import com.lz.mlengine.core.tree._
import org.apache.spark.ml.tree.{InternalNode => SparkInternalNode, LeafNode => SparkLeafNode, Node => SparkNode}
import org.apache.spark.ml.tree.{CategoricalSplit => SparkCategoricalSplit, Split => SparkSplit}
import org.apache.spark.ml.tree.{ContinuousSplit => SparkContinuousSplit}
import org.apache.spark.mllib.tree.impurity.ImpurityCalculator

object TreeConverter {

  implicit def convert(split: SparkSplit): Split = {
    split match {
      case s: SparkContinuousSplit => new ContinuousSplit(s.featureIndex, s.threshold)
      case s: SparkCategoricalSplit => new DiscreteSplit(s.featureIndex, s.leftCategories.toSet)
    }
  }

  implicit def convert(impurityStats: ImpurityCalculator): Vector[Double] = {
    DenseVector[Double](impurityStats.stats)
  }

  def convertDecisionTree(node: SparkNode): Node = {
    node match {
      case n: SparkInternalNode =>
        ImpurityCalculator
        new InternalNode(
          convertDecisionTree(n.leftChild),
          convertDecisionTree(n.rightChild),
          n.split,
          n.prediction,
          n.impurity,
          n.impurityStats
        )
      case n: SparkLeafNode =>
        new LeafNode(n.prediction, n.impurity, n.impurityStats)
    }
  }

}
