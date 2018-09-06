package com.lz.mlengine.spark

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.{classification => cl}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{regression => rg}
import org.scalatest.{FlatSpec, Matchers}
import com.lz.mlengine.core.FeatureSet
import com.lz.mlengine.core.classification.LogisticRegressionModel
import com.lz.mlengine.core.regression.LinearRegressionModel

import scala.collection.mutable.{Map => MutableMap}
import scala.concurrent.Await
import scala.concurrent.duration._

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

class TrainerObjectTest extends FlatSpec with Matchers with DatasetSuiteBase {

  import spark.implicits._

  "train" should "train and evaluate classification model" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 1.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("4", MutableMap("feature1" -> 1.0, "feature2" -> 1.0)),
      FeatureSet("5", MutableMap("feature2" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("6", MutableMap("feature1" -> 1.0, "feature3" -> 1.0))
    ).toDS
    val trainLabels = Seq(("1", "a"), ("2", "b"), ("3", "a")).toDS
    val testLabels = Seq(("4", "a"), ("5", "b"), ("6", "a")).toDS
    val trainer = new ClassificationTrainer[cl.LogisticRegression, cl.LogisticRegressionModel](
      new cl.LogisticRegression())(spark)

    val (model, metrics) = Trainer.train(trainer, features, trainLabels, testLabels)(spark)
    model.asInstanceOf[LogisticRegressionModel].coefficients.size should be (3)
    metrics.confusionMatrices.size should be (2)
  }

  "train" should "train and evaluate regression model" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 1.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("4", MutableMap("feature1" -> 1.0, "feature2" -> 1.0)),
      FeatureSet("5", MutableMap("feature2" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("6", MutableMap("feature1" -> 1.0, "feature3" -> 1.0))
    ).toDS
    val trainLabels = Seq(("1", 0.8), ("2", 0.5), ("3", 0.2)).toDS
    val testLabels = Seq(("4", 0.8), ("5", 0.5), ("6", 0.2)).toDS
    val trainer = new RegressionTrainer[rg.LinearRegression, rg.LinearRegressionModel](
      new rg.LinearRegression())(spark)

    val (model, metrics) = Trainer.train(trainer, features, trainLabels, testLabels)(spark)
    model.asInstanceOf[LinearRegressionModel].coefficients.size should be (3)
    metrics.explainedVariance should be >= -0.00001
    metrics.meanAbsoluteError should be >= -0.00001
    metrics.meanSquaredError should be >= -0.00001
    metrics.r2 should be >= -0.00001
  }

  "trainMultipleClassifier" should "train and evaluate multiple classification models" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 1.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("4", MutableMap("feature1" -> 1.0, "feature2" -> 1.0)),
      FeatureSet("5", MutableMap("feature2" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("6", MutableMap("feature1" -> 1.0, "feature3" -> 1.0))
    ).toDS
    val trainLabels = Seq(("1", "a"), ("2", "b"), ("3", "a")).toDS
    val testLabels = Seq(("4", "a"), ("5", "b"), ("6", "a")).toDS
    val trainers = Seq(
      new ClassificationTrainer[cl.LogisticRegression, cl.LogisticRegressionModel](
        new cl.LogisticRegression().setRegParam(0.1))(spark),
      new ClassificationTrainer[cl.LogisticRegression, cl.LogisticRegressionModel](
        new cl.LogisticRegression().setRegParam(0.01))(spark),
      new ClassificationTrainer[cl.LogisticRegression, cl.LogisticRegressionModel](
        new cl.LogisticRegression().setRegParam(0.001))(spark)
    )

    val results = Trainer.trainMultipleClassifier(trainers, features, trainLabels, testLabels)(spark)
    results.foreach { f =>
      val (model, metrics) = Await.result(f, 100.second)
      model.asInstanceOf[LogisticRegressionModel].coefficients.size should be (3)
      metrics.confusionMatrices.size should be (2)
    }
  }

  "trainMultipleRegressor" should "train and evaluate multiple regression models" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 1.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("4", MutableMap("feature1" -> 1.0, "feature2" -> 1.0)),
      FeatureSet("5", MutableMap("feature2" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("6", MutableMap("feature1" -> 1.0, "feature3" -> 1.0))
    ).toDS
    val trainLabels = Seq(("1", 0.8), ("2", 0.5), ("3", 0.2)).toDS
    val testLabels = Seq(("4", 0.8), ("5", 0.5), ("6", 0.2)).toDS
    val trainers = Seq(
      new RegressionTrainer[rg.LinearRegression, rg.LinearRegressionModel](
        new rg.LinearRegression().setRegParam(0.1))(spark),
      new RegressionTrainer[rg.LinearRegression, rg.LinearRegressionModel](
        new rg.LinearRegression().setRegParam(0.01))(spark),
      new RegressionTrainer[rg.LinearRegression, rg.LinearRegressionModel](
        new rg.LinearRegression().setRegParam(0.001))(spark)
    )

    val results = Trainer.trainMultipleRegressor(trainers, features, trainLabels, testLabels)(spark)
    results.foreach { f =>
      val (model, metrics) = Await.result(f, 100.second)
      model.asInstanceOf[LinearRegressionModel].coefficients.size should be (3)
      metrics.explainedVariance should be >= -0.00001
      metrics.meanAbsoluteError should be >= -0.00001
      metrics.meanSquaredError should be >= -0.00001
      metrics.r2 should be >= -0.00001
    }
  }

}