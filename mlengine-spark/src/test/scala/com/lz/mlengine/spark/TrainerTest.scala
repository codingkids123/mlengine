package com.lz.mlengine.spark

import scala.collection.mutable.{Map => MutableMap}

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.{classification => cl}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{regression => rg}
import org.scalatest.{FlatSpec, Matchers}

import com.lz.mlengine.core.FeatureSet
import com.lz.mlengine.core.classification.LogisticRegressionModel
import com.lz.mlengine.core.regression.LinearRegressionModel

class ClassificationTrainerTest extends FlatSpec with Matchers with DatasetSuiteBase {

  import spark.implicits._

  "getFeatureToIndexMap" should "generate a map from feature name to index" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 1.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 1.0))
    ).toDS

    val featureToIndexMap = getTrainer().getFeatureToIndexMap(features)

    featureToIndexMap should be(Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2))
  }

  "getLabelToIndexMap" should "generate a map from label to index" in {
    val labels = Seq(
      ("1", "positive"),
      ("2", "positive"),
      ("3", "negative")
    ).toDS

    val t = getTrainer()
    val labelToIndexMap = t.getLabelToIndexMap(labels)

    labelToIndexMap should be(Map("negative" -> 0, "positive" -> 1))
  }

  "getLabeledSparkFeature" should "map string valued prediction to label with label to index map" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 1.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 1.0))
    ).toDS

    val labels = Seq(("1", "positive"), ("2", "positive"), ("3", "negative")).toDS

    val featureToIndexMap = Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2)
    val labelToIndexMap = Map("negative" -> 0, "positive" -> 1)

    val labeledFeatures = getTrainer().getLabeledSparkFeature(features, labels, featureToIndexMap, labelToIndexMap)

    val expected = Seq(
      LabeledSparkFeature("1", Vectors.sparse(3, Seq((0, 1.0), (1, 1.0))), 1.0),
      LabeledSparkFeature("2", Vectors.sparse(3, Seq((1, 1.0), (2, 1.0))), 1.0),
      LabeledSparkFeature("3", Vectors.sparse(3, Seq((0, 1.0), (2, 1.0))), 0.0)
    ).toDS

    assertDatasetEquals(expected, labeledFeatures)
  }

  "fit" should "train classification model with index to label map" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 1.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 1.0))
    ).toDS

    val labels = Seq(("1", "positive"), ("2", "positive"), ("3", "negative")).toDS

    val trainer = getTrainer()
    val model = trainer.fit(features, labels).asInstanceOf[LogisticRegressionModel]

    model.coefficients.rows should be (1)
    model.coefficients.cols should be (3)
    model.intercept.size should be (1)
    model.featureToIndexMap should be (Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2))
    model.indexToLabelMap should be (Map(0 -> "negative", 1 -> "positive"))
  }

  def getTrainer(): ClassificationTrainer[cl.LogisticRegression, cl.LogisticRegressionModel] = {
    new ClassificationTrainer[cl.LogisticRegression, cl.LogisticRegressionModel](new cl.LogisticRegression())(spark)
  }

}

class RegressionTrainerTest extends FlatSpec with Matchers with DatasetSuiteBase {

  import spark.implicits._

  "getLabeledSparkFeature" should "map double valued prediction to label without label to index map" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 1.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 1.0))
    ).toDS

    val labels = Seq(("1", 0.8), ("2", 0.5), ("3", 0.2)).toDS

    val featureToIndexMap = Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2)

    val labeledFeatures = getTrainer()
      .getLabeledSparkFeature(features, labels, featureToIndexMap)

    val expected = Seq(
      LabeledSparkFeature("1", Vectors.sparse(3, Seq((0, 1.0), (1, 1.0))), 0.8),
      LabeledSparkFeature("2", Vectors.sparse(3, Seq((1, 1.0), (2, 1.0))), 0.5),
      LabeledSparkFeature("3", Vectors.sparse(3, Seq((0, 1.0), (2, 1.0))), 0.2)
    ).toDS

    assertDatasetEquals(expected, labeledFeatures)
  }

  "fit" should "train regression model without index to label map" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 1.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 1.0))
    ).toDS

    val labels = Seq(("1", 0.8), ("2", 0.5), ("3", 0.2)).toDS

    val trainer = getTrainer()
    val model = trainer.fit(features, labels).asInstanceOf[LinearRegressionModel]

    model.coefficients.size should be (3)
    model.featureToIndexMap should be(Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2))
  }

  def getTrainer(): RegressionTrainer[rg.LinearRegression, rg.LinearRegressionModel] = {
    new RegressionTrainer[rg.LinearRegression, rg.LinearRegressionModel](new rg.LinearRegression())(spark)
  }

}
