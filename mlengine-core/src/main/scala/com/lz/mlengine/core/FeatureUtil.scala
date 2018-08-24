package com.lz.mlengine.core

object FeatureUtil {

  def addCategoricalFeature(name: String, featureSet: FeatureSet, value: String): Unit = {
    featureSet.features += (s"$name:$value" -> 1.0)
  }

  def addKeyValueFeature(name: String, featureSet: FeatureSet, values: Map[String, Double]): Unit = {
    featureSet.features ++= values.map { case(k, v) => (s"$name:$k", v) }
  }

  def addScalarFeature(name: String, featureSet: FeatureSet, value: Double): Unit = {
    featureSet.features += (name -> value)
  }

  def addVectorFeature(name: String, featureSet: FeatureSet, values: Seq[Double]): Unit = {
    featureSet.features ++= values.zipWithIndex.map { case(v, i) => (s"$name:$i", v) }
  }

  def addSparseFeature(name: String, featureSet: FeatureSet, values: Map[Int, Double]): Unit = {
    featureSet.features ++= values.map { case(k, v) => (s"$name:$k", v) }
  }

}