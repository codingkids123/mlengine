package com.lz.mlengine

import org.apache.spark.ml.Model
import org.apache.spark.ml.classification._
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

  def decisionTreeClassificationModel(path: String)(implicit spark: SparkSession) = {
    load[DecisionTreeClassificationModel](DecisionTreeClassificationModel.load, path)
  }

  def gBTClassificationModel(path: String)(implicit spark: SparkSession) = {
    load[GBTClassificationModel](GBTClassificationModel.load, path)
  }

  def linearSVCModel(path: String)(implicit spark: SparkSession) = {
    load[LinearSVCModel](LinearSVCModel.load, path)
  }

  def logisticRegressionModel(path: String)(implicit spark: SparkSession) = {
    load[LogisticRegressionModel](LogisticRegressionModel.load, path)
  }

  def multilayerPerceptronClassificationModel(path: String)(implicit spark: SparkSession) = {
    load[MultilayerPerceptronClassificationModel](MultilayerPerceptronClassificationModel.load, path)
  }

  def naiveBayesModel(path: String)(implicit spark: SparkSession) = {
    load[NaiveBayesModel](NaiveBayesModel.load, path)
  }

  def randomForestClassificationModel(path: String)(implicit spark: SparkSession) = {
    load[RandomForestClassificationModel](RandomForestClassificationModel.load, path)
  }

}
