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

    val sparkFeatures = getModel().getSparkFeatures(features)

    val expected = Seq(
      SparkFeature("1", Vectors.sparse(3, Seq((0, 1.0), (1, 0.0)))),
      SparkFeature("2", Vectors.sparse(3, Seq((1, 1.0), (2, 0.0)))),
      SparkFeature("3", Vectors.sparse(3, Seq((0, 1.0), (2, 0.0))))
    ).toDS

    assertDatasetEquals(expected, sparkFeatures)
  }

  "getSparkFeatures" should "ignore feature not present in featureToIndexMap" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 0.0, "feature4" -> 0.0))
    ).toDS

    val sparkFeatures = getModel().getSparkFeatures(features)

    val expected = Seq(
      SparkFeature("1", Vectors.sparse(3, Seq((0, 1.0), (1, 0.0))))
    ).toDS

    assertDatasetEquals(expected, sparkFeatures)
  }

  "getPredictionSets" should "convert SparkPrediction to PredictionSet" in {
    val sparkPredictions = Seq(
      SparkPrediction("1", Vectors.dense(Array(0.2, 0.8)), Vectors.dense(Array(-0.8, 0.8)), 1.0),
      SparkPrediction("2", Vectors.dense(Array(0.3, 0.7)), Vectors.dense(Array(-0.7, 0.7)), 1.0),
      SparkPrediction("3", Vectors.dense(Array(0.8, 0.2)), Vectors.dense(Array(0.8, -0.8)), 0.0)
    ).toDS

    val predictions = getModel().getPredictionSets(sparkPredictions)

    val expected = Seq(
      PredictionSet("1", Seq(Prediction(Some("negative"), Some(0.2)), Prediction(Some("positive"), Some(0.8)))),
      PredictionSet("2", Seq(Prediction(Some("negative"), Some(0.3)), Prediction(Some("positive"), Some(0.7)))),
      PredictionSet("3", Seq(Prediction(Some("negative"), Some(0.8)), Prediction(Some("positive"), Some(0.2))))
    ).toDS

    assertDatasetEquals(expected, predictions)
  }

  "predict" should "output prediction from input feature set" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 0.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 0.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 0.0))
    ).toDS

    val predictions = getModel().predict(features)

    val expected = Seq(
      PredictionSet("1", Seq(Prediction(Some("negative"), Some(0.2)), Prediction(Some("positive"), Some(0.8)))),
      PredictionSet("2", Seq(Prediction(Some("negative"), Some(0.3)), Prediction(Some("positive"), Some(0.7)))),
      PredictionSet("3", Seq(Prediction(Some("negative"), Some(0.8)), Prediction(Some("positive"), Some(0.2))))
    ).toDS

    assertDatasetEquals(expected, predictions)
  }

  def getModel() = {
    val predictions = Seq(
      SparkPrediction("1", Vectors.dense(Array(0.2, 0.8)), Vectors.dense(Array(-0.8, 0.8)), 1.0),
      SparkPrediction("2", Vectors.dense(Array(0.3, 0.7)), Vectors.dense(Array(-0.7, 0.7)), 1.0),
      SparkPrediction("3", Vectors.dense(Array(0.8, 0.2)), Vectors.dense(Array(0.8, -0.8)), 0.0)
    ).toDS
    new SparkModel[MockSparkModel](
      new MockSparkModel(predictions),
      Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2), Map(0 -> "negative", 1 -> "positive")
    )(spark)
  }

}
