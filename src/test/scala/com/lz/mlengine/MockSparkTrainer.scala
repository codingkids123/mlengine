package com.lz.mlengine

import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.types.StructType

class MockSparkTrainer(model: MockSparkModel)(implicit spark: SparkSession) extends Estimator[MockSparkModel]{

  import spark.implicits._

  var trainingData: Dataset[LabeledSparkFeature] = _

  override def fit(dataset: Dataset[_]): MockSparkModel = {
    trainingData = dataset.as[LabeledSparkFeature]
    model
  }

  override def copy(extra: ParamMap): Estimator[MockSparkModel] = ???

  override def transformSchema(schema: StructType): StructType = ???

  override val uid: String = ""

}
