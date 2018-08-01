package com.lz.mlengine

import org.apache.spark.ml.Model
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{MLWritable, MLWriter}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

class MockSparkModel extends Model[MockSparkModel] with MLWritable {

  override def copy(extra: ParamMap): Nothing = ???

  override def transform(dataset: Dataset[_]): DataFrame = ???

  override def transformSchema(schema: StructType): StructType = ???

  override val uid: String = ""

  override def write: MLWriter = ???

}
