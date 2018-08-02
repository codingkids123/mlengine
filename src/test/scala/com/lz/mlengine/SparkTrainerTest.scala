package com.lz.mlengine

import scala.collection.mutable.{Map => MutableMap}
import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.scalatest.{FlatSpec, Matchers}

class SparkTrainerTest extends FlatSpec with Matchers with DatasetSuiteBase {

  import spark.implicits._

  "getFeatureToIndexMap" should "generate a map from feature name to index" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 0.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 0.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 0.0))
    ).toDS

    val featureToIndexMap = getClassificationTrainer().getFeatureToIndexMap(features)

    featureToIndexMap should be(Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2))
  }

  "getLabelToIndexMap" should "generate a map from label to index" in {
    val labels = Seq(
      PredictionSet("1", Map("positive" -> 1.0)),
      PredictionSet("2", Map("positive" -> 1.0)),
      PredictionSet("3", Map("negative" -> 1.0))
    ).toDS

    val labelToIndexMap = getClassificationTrainer().getLabelToIndexMap(labels)

    labelToIndexMap should be(Map("negative" -> 0, "positive" -> 1))
  }

  "getLabeledSparkFeature" should "map string valued prediction to label with label to index map" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 0.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 0.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 0.0))
    ).toDS

    val labels = Seq(
      PredictionSet("1", Map("positive" -> 1.0)),
      PredictionSet("2", Map("positive" -> 1.0)),
      PredictionSet("3", Map("negative" -> 1.0))
    ).toDS

    val featureToIndexMap = Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2)
    val labelToIndexMap = Map("negative" -> 0, "positive" -> 1)

    val labeledFeatures = getClassificationTrainer()
      .getLabeledSparkFeature(features, labels, featureToIndexMap, Some(labelToIndexMap))

    val expected = Seq(
      LabeledSparkFeature("1", Vectors.sparse(3, Seq((0, 1.0), (1, 0.0))), 1.0),
      LabeledSparkFeature("2", Vectors.sparse(3, Seq((1, 1.0), (2, 0.0))), 1.0),
      LabeledSparkFeature("3", Vectors.sparse(3, Seq((0, 1.0), (2, 0.0))), 0.0)
    ).toDS

    assertDatasetEquals(expected, labeledFeatures)
  }

  it should "map double valued prediction to label without label to index map" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 0.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 0.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 0.0))
    ).toDS

    val labels = Seq(
      PredictionSet("1", Map("value" -> 0.8)),
      PredictionSet("2", Map("value" -> 0.5)),
      PredictionSet("3", Map("value" -> 0.2))
    ).toDS

    val featureToIndexMap = Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2)
    val labelToIndexMap = Map("negative" -> 0, "positive" -> 1)

    val labeledFeatures = getRegressionTrainer()
      .getLabeledSparkFeature(features, labels, featureToIndexMap, None)

    val expected = Seq(
      LabeledSparkFeature("1", Vectors.sparse(3, Seq((0, 1.0), (1, 0.0))), 0.8),
      LabeledSparkFeature("2", Vectors.sparse(3, Seq((1, 1.0), (2, 0.0))), 0.5),
      LabeledSparkFeature("3", Vectors.sparse(3, Seq((0, 1.0), (2, 0.0))), 0.2)
    ).toDS

    assertDatasetEquals(expected, labeledFeatures)
  }

  "fit" should "train classification model with index to label map" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 0.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 0.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 0.0))
    ).toDS

    val labels = Seq(
      PredictionSet("1", Map("positive" -> 1.0)),
      PredictionSet("2", Map("positive" -> 1.0)),
      PredictionSet("3", Map("negative" -> 1.0))
    ).toDS

    val trainer = getClassificationTrainer()
    val model = trainer.fit(features, Some(labels))

    model.model should not be(null)
    model.featureToIndexMap should be(Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2))
    model.indexToLabelMapMaybe.get should be(Map(0 -> "negative", 1 -> "positive"))
  }

  "fit" should "train regression model without index to label map" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 0.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 0.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 0.0))
    ).toDS

    val labels = Seq(
      PredictionSet("1", Map("value" -> 0.8)),
      PredictionSet("2", Map("value" -> 0.5)),
      PredictionSet("3", Map("value" -> 0.2))
    ).toDS

    val trainer = getRegressionTrainer()
    val model = trainer.fit(features, Some(labels))

    model.model should not be(null)
    model.featureToIndexMap should be(Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2))
    model.indexToLabelMapMaybe should be(None)
  }

  "fit" should "train clustering model without labels" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 0.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 0.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 0.0))
    ).toDS

    val trainer = getClusteringTrainer()
    val model = trainer.fit(features, None)

    model.model should not be(null)
    model.featureToIndexMap should be(Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2))
    model.indexToLabelMapMaybe should be(None)
  }

  def getClassificationTrainer() = {
    new SparkTrainer[LogisticRegression, LogisticRegressionModel](new LogisticRegression())(spark)
  }

  def getRegressionTrainer() = {
    new SparkTrainer[LinearRegression, LinearRegressionModel](new LinearRegression())(spark)
  }

  def getClusteringTrainer() = {
    new SparkTrainer[KMeans, KMeansModel](new KMeans())(spark)
  }

}
