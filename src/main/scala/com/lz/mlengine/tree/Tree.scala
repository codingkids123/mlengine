package com.lz.mlengine.tree

import breeze.linalg.Vector

trait Node extends Serializable {

  val impurityStats: Vector[Double]

  def predictImpl(feature: Vector[Double]): Node

}

class InternalNode(val left: Node, val right: Node, val split: Split, val impurityStats: Vector[Double]) extends Node {

  def predictImpl(feature: Vector[Double]): Node = {
    if (split.shouldGoLeft(feature)) {
      left.predictImpl(feature)
    } else {
      right.predictImpl(feature)
    }
  }

}

class LeafNode(val prediction: Double, val impurity: Double, val impurityStats: Vector[Double]) extends Node {

  def predictImpl(feature: Vector[Double]): Node = this

}

trait Split extends Serializable {

  def shouldGoLeft(feature: Vector[Double]): Boolean

}

class ContinuousSplit(val featureIndex: Int, val threshold: Double) extends Split {

  override def shouldGoLeft(feature: Vector[Double]): Boolean = {
    feature(featureIndex) < threshold
  }

}

class DiscreteSplit( val featureIndex: Int, val leftValueSet: Set[Double]) extends Split {

  override def shouldGoLeft(feature: Vector[Double]): Boolean = {
    leftValueSet.contains(feature(featureIndex))
  }

}

class Tree(val root: Node) extends Serializable {

  def predict(feature: Vector[Double]): Vector[Double] = {
    root.predictImpl(feature).impurityStats
  }

}
