package com.lz.mlengine

import org.apache.spark.sql.types._

case class PredictionSet(id: String, predictions: Map[String, Double])

object PredictionSet {
  def schema = StructType(Seq(
    StructField("id", StringType), StructField("predictions", MapType(StringType, DoubleType))
  ))
}