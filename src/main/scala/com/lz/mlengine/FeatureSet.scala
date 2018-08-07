package com.lz.mlengine

import scala.collection.mutable.Map

import org.apache.spark.sql.types._

case class FeatureSet(id: String, features: Map[String, Double] = Map())

object FeatureSet {
  def schema = StructType(Seq(
    StructField("id", StringType), StructField("features", MapType(StringType, DoubleType))
  ))
}