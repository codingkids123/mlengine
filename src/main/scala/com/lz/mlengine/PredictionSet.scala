package com.lz.mlengine

import org.apache.spark.sql.types._

case class Prediction(label: Option[String], value: Option[Double])

case class PredictionSet(id: String, predictions: Seq[Prediction])

object PredictionSet {
  def schema = StructType(Seq(
    StructField("id", StringType),
    StructField("predictions", ArrayType(StructType(Seq(
      StructField("label", StringType, true),
      StructField("value", DoubleType)))))
  ))
}