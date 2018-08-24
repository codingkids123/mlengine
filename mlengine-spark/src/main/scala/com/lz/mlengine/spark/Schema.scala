package com.lz.mlengine.spark

import org.apache.spark.sql.types._

object Schema {

  def featureSet = StructType(Seq(
    StructField("id", StringType), StructField("features", MapType(StringType, DoubleType))
  ))


  def predictionSet = StructType(Seq(
    StructField("id", StringType), StructField("predictions", MapType(StringType, DoubleType))
  ))

}
