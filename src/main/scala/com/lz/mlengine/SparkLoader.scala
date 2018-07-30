package com.lz.mlengine

import org.apache.spark.ml.Model
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.SparkSession

object SparkLoader {

  def load[M <: Model[M] with MLWritable](modelLoadFunc: (String) => M, path: String)
                                         (implicit spark: SparkSession): SparkModel[M] = {
    import spark.implicits._

    val model = modelLoadFunc(s"${path}/model")
    val featureToIndexMap = spark.read.format("parquet").load(s"${path}/feature_to_idx").as[(String, Int)].collect.toMap
    val indexToLabelMap = spark.read.format("parquet").load(s"${path}/idx_to_label").as[(Int, String)].collect.toMap

    new SparkModel[M](model, featureToIndexMap, indexToLabelMap)
  }

  def logisticRegressionModel(path: String)(implicit spark: SparkSession) = {
    load[LogisticRegressionModel](LogisticRegressionModel.load, path)
  }

  def decisionTreeClassificationModel(path: String)(implicit spark: SparkSession) = {
    load[DecisionTreeClassificationModel](DecisionTreeClassificationModel.load, path)
  }

}
