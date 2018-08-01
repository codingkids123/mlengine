package com.lz.mlengine

import org.apache.spark.sql.types._

import scala.collection.mutable.Map

case class FeatureSet(id: String, features: Map[String, Double] = Map())

object FeatureSet {
  def schema = StructType(Seq(
    StructField("id", StringType), StructField("features", MapType(StringType, DoubleType))
  ))
}