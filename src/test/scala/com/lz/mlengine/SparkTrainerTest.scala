package com.lz.mlengine

import scala.collection.mutable.{Map => MutableMap}
import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.{FlatSpec, Matchers}

class SparkTrainerTest extends FlatSpec with Matchers with DatasetSuiteBase {

  import spark.implicits._

  val sparkModel = new MockSparkModel(null)

  "getFeatureToIndexMap" should "generate a map from feature name to index" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 0.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 0.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 0.0))
    ).toDS

    val featureToIndexMap = getTrainer().getFeatureToIndexMap(features)

    featureToIndexMap should be (Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2))
  }

  "getLabelToIndexMap" should "generate a map from label to index" in {
    val labels = Seq(
      PredictionSet("1", Seq(Prediction(Some("positive"), None))),
      PredictionSet("2", Seq(Prediction(Some("positive"), None))),
      PredictionSet("3", Seq(Prediction(Some("negative"), None)))
    ).toDS

    val labelToIndexMap = getTrainer().getLabelToIndexMap(labels)

    labelToIndexMap should be (Map("negative" -> 0, "positive" -> 1))
  }

  "getLabeledFeature" should "join feature and label by id" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 0.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 0.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 0.0))
    ).toDS

    val labels = Seq(
      PredictionSet("1", Seq(Prediction(Some("positive"), None))),
      PredictionSet("2", Seq(Prediction(Some("positive"), None))),
      PredictionSet("3", Seq(Prediction(Some("negative"), None)))
    ).toDS

    val featureToIndexMap = Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2)
    val labelToIndexMap = Map("negative" -> 0, "positive" -> 1)

    val labeledFeatures = getTrainer().generateLabeledSparkFeature(features, labels, featureToIndexMap, labelToIndexMap)

    val expected = Seq(
      LabeledSparkFeature("1", Vectors.sparse(3, Seq((0, 1.0), (1, 0.0))), 1.0),
      LabeledSparkFeature("2", Vectors.sparse(3, Seq((1, 1.0), (2, 0.0))), 1.0),
      LabeledSparkFeature("3", Vectors.sparse(3, Seq((0, 1.0), (2, 0.0))), 0.0)
    ).toDS

    assertDatasetEquals(expected, labeledFeatures)
  }

  "fit" should "train model use features and labels" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 0.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 0.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 0.0))
    ).toDS

    val labels = Seq(
      PredictionSet("1", Seq(Prediction(Some("positive"), None))),
      PredictionSet("2", Seq(Prediction(Some("positive"), None))),
      PredictionSet("3", Seq(Prediction(Some("negative"), None)))
    ).toDS

    val trainer = getTrainer()
    val model = trainer.fit(features, labels)

    model.model should be (sparkModel)
    model.featureToIndexMap should be (Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2))
    model.indexToLabelMap should be (Map(0 -> "negative", 1 -> "positive"))

    val expected = Seq(
      LabeledSparkFeature("1", Vectors.sparse(3, Seq((0, 1.0), (1, 0.0))), 1.0),
      LabeledSparkFeature("2", Vectors.sparse(3, Seq((1, 1.0), (2, 0.0))), 1.0),
      LabeledSparkFeature("3", Vectors.sparse(3, Seq((0, 1.0), (2, 0.0))), 0.0)
    ).toDS

    assertDatasetEquals(expected, trainer.trainer.trainingData)
  }

  def getTrainer() = {
    new SparkTrainer[MockSparkTrainer, MockSparkModel](new MockSparkTrainer(sparkModel)(spark))(spark)
  }

}
