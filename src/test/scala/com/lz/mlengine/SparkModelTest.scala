package com.lz.mlengine

import scala.collection.mutable.{Map => MutableMap}
import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.linalg.Vectors
import org.scalamock.proxy.ProxyMockFactory
import org.scalatest.{FlatSpec, Matchers}

class SparkModelTest extends FlatSpec with Matchers with DatasetSuiteBase with ProxyMockFactory {

  import spark.implicits._

  "getSparkFeatures" should "convert FeatureSet to SparkFeature" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 0.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 0.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 0.0))
    ).toDS

    val sparkFeatures = getClassificationModel().getSparkFeatures(features)

    val expected = Seq(
      SparkFeature("1", Vectors.sparse(3, Seq((0, 1.0), (1, 0.0)))),
      SparkFeature("2", Vectors.sparse(3, Seq((1, 1.0), (2, 0.0)))),
      SparkFeature("3", Vectors.sparse(3, Seq((0, 1.0), (2, 0.0))))
    ).toDS

    assertDatasetEquals(expected, sparkFeatures)
  }

  it should "ignore feature not present in featureToIndexMap" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 0.0, "feature4" -> 0.0))
    ).toDS

    val sparkFeatures = getClassificationModel().getSparkFeatures(features)

    val expected = Seq(
      SparkFeature("1", Vectors.sparse(3, Seq((0, 1.0), (1, 0.0))))
    ).toDS

    assertDatasetEquals(expected, sparkFeatures)
  }

  "getPredictionSets" should "convert SparkPredictionVector in classification model" in {
    val sparkPredictions = Seq(
      SparkPredictionVector("1", Vectors.dense(Array(-0.8, 0.8))),
      SparkPredictionVector("2", Vectors.dense(Array(-0.5, 0.5))),
      SparkPredictionVector("3", Vectors.dense(Array(-0.2, 0.2)))
    ).toDF("id", "rawPrediction")

    val predictions = getClassificationModel().getPredictionSets(sparkPredictions)

    val expected = Seq(
      PredictionSet("1", Seq(Prediction(Some("negative"), Some(-0.8)), Prediction(Some("positive"), Some(0.8)))),
      PredictionSet("2", Seq(Prediction(Some("negative"), Some(-0.5)), Prediction(Some("positive"), Some(0.5)))),
      PredictionSet("3", Seq(Prediction(Some("negative"), Some(-0.2)), Prediction(Some("positive"), Some(0.2))))
    ).toDS

    assertDatasetEquals(expected, predictions)
  }

  it should "convert SparkPredictionScalar in regression model" in {
    val sparkPredictions = Seq(
      SparkPredictionScalar("1", 0.8),
      SparkPredictionScalar("2", 0.5),
      SparkPredictionScalar("3", 0.2)
    ).toDF("id", "prediction")

    val predictions = getRegressionModel().getPredictionSets(sparkPredictions)

    val expected = Seq(
      PredictionSet("1", Seq(Prediction(None, Some(0.8)))),
      PredictionSet("2", Seq(Prediction(None, Some(0.5)))),
      PredictionSet("3", Seq(Prediction(None, Some(0.2))))
    ).toDS

    assertDatasetEquals(expected, predictions)
  }

  def getClassificationModel() = {
    new SparkModel[MockSparkModel](
      new MockSparkModel(),
      Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2), Some(Map(0 -> "negative", 1 -> "positive"))
    )(spark)
  }

  def getRegressionModel() = {
    new SparkModel[MockSparkModel](
      new MockSparkModel(),
      Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2), None
    )(spark)
  }

}
