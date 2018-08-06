package com.lz.mlengine

import org.apache.spark.ml.Model
import org.apache.spark.ml.classification
import org.apache.spark.ml.clustering
import org.apache.spark.ml.regression
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.SparkSession

object SparkLoader {

  def load[M <: Model[M] with MLWritable](modelLoadFunc: (String) => M, path: String,
                                          loadIndexToLabelMap: Boolean = false)
                                         (implicit spark: SparkSession): SparkModel[M] = {
    import spark.implicits._

    val model = modelLoadFunc(s"${path}/model")
    val featureToIndexMap = spark.read.format("parquet").load(s"${path}/feature_to_idx").as[(String, Int)].collect.toMap
    if (loadIndexToLabelMap) {
      val indexToLabelMap = spark.read.format("parquet").load(s"${path}/idx_to_label").as[(Int, String)].collect.toMap
      new SparkModel[M](model, featureToIndexMap, Some(indexToLabelMap))
    } else {
      new SparkModel[M](model, featureToIndexMap, None)
    }
  }

  // Classification models.
  def decisionTreeClassificationModel(path: String)(implicit spark: SparkSession) = {
    load[classification.DecisionTreeClassificationModel](
      classification.DecisionTreeClassificationModel.load, path,true
    )
  }

  def gBTClassificationModel(path: String)(implicit spark: SparkSession) = {
    load[classification.GBTClassificationModel](classification.GBTClassificationModel.load, path, true)
  }

  def linearSVCModel(path: String)(implicit spark: SparkSession) = {
    load[classification.LinearSVCModel](classification.LinearSVCModel.load, path, true)
  }

  def logisticRegressionModel(path: String)(implicit spark: SparkSession) = {
    load[classification.LogisticRegressionModel](classification.LogisticRegressionModel.load, path, true)
  }

  def multilayerPerceptronClassificationModel(path: String)(implicit spark: SparkSession) = {
    load[classification.MultilayerPerceptronClassificationModel](
      classification.MultilayerPerceptronClassificationModel.load, path
    )
  }

  def naiveBayesModel(path: String)(implicit spark: SparkSession) = {
    load[classification.NaiveBayesModel](classification.NaiveBayesModel.load, path, true)
  }

  def randomForestClassificationModel(path: String)(implicit spark: SparkSession) = {
    load[classification.RandomForestClassificationModel](
      classification.RandomForestClassificationModel.load, path, true
    )
  }

  // Regression models.
  def aFTSurvivalRegressionModel(path: String)(implicit spark: SparkSession) = {
    load[regression.AFTSurvivalRegressionModel](regression.AFTSurvivalRegressionModel.load, path, false)
  }

  def decisionTreeRegressorModel(path: String)(implicit spark: SparkSession) = {
    load[regression.DecisionTreeRegressionModel](regression.DecisionTreeRegressionModel.load, path, false)
  }
  def gBTRegressionModel(path: String)(implicit spark: SparkSession) = {
    load[regression.GBTRegressionModel](regression.GBTRegressionModel.load, path, false)
  }
  def generalizedLinearRegressionModel(path: String)(implicit spark: SparkSession) = {
    load[regression.GeneralizedLinearRegressionModel](regression.GeneralizedLinearRegressionModel.load, path, false)
  }

  def isotonicRegressionModel(path: String)(implicit spark: SparkSession) = {
    load[regression.IsotonicRegressionModel](regression.IsotonicRegressionModel.load, path, false)
  }

  def linearRegressionModel(path: String)(implicit spark: SparkSession) = {
    load[regression.LinearRegressionModel](regression.LinearRegressionModel.load, path, false)
  }

  def randomForestRegressionModel(path: String)(implicit spark: SparkSession) = {
    load[regression.RandomForestRegressionModel](regression.RandomForestRegressionModel.load, path, false)
  }

  // Clustering models.
  def kMeansModel(path: String)(implicit spark: SparkSession) = {
    load[clustering.KMeansModel](clustering.KMeansModel.load, path, false)
  }

  def gaussianMixtureModel(path: String)(implicit spark: SparkSession) = {
    load[clustering.GaussianMixtureModel](clustering.GaussianMixtureModel.load, path, false)
  }


}
