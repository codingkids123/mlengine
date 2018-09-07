package com.lz.mlengine.spark

import scala.collection.mutable.{Map => MutableMap}

import breeze.linalg
import breeze.linalg.DenseVector
import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.scalatest.{FlatSpec, Matchers}

import com.lz.mlengine.core.{ClassificationModel, FeatureSet, RegressionModel}

class MockClassificationModel(predictFun: (linalg.Vector[Double]) => linalg.Vector[Double],
                              override val featureToIndexMap: Map[String, Int],
                              override val indexToLabelMap: Map[Int, String])
  extends ClassificationModel(featureToIndexMap, indexToLabelMap) {

  override private[mlengine] def predictImpl(vector: linalg.Vector[Double]) = {
    predictFun(vector)
  }

}

class MockRegressionModel(predictFun: (linalg.Vector[Double]) => linalg.Vector[Double],
                          override val featureToIndexMap: Map[String, Int])
  extends RegressionModel(featureToIndexMap) {

  override private[mlengine] def predictImpl(vector: linalg.Vector[Double]) = {
    predictFun(vector)
  }

}

class EvaluatorTest extends FlatSpec with Matchers with DatasetSuiteBase {

  import spark.implicits._

  "predict" should "predict and join labels for classification" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 1.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("4", MutableMap("feature1" -> 1.0, "feature2" -> 1.0)),
      FeatureSet("5", MutableMap("feature2" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("6", MutableMap("feature1" -> 1.0, "feature3" -> 1.0))
    ).toDS
    val labels = Seq(("1", "a"), ("2", "b"), ("3", "a")).toDS
    val model = new MockClassificationModel(
      v => DenseVector(v(0), v(1)), Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2), Map(0 -> "a", 1 -> "b")
    )

    val predictions = Evaluator.predict(features, labels, model)(spark)
    val expected = Seq(
      (Map("a" -> 1.0, "b" -> 1.0), "a"),
      (Map("a" -> 0.0, "b" -> 1.0), "b"),
      (Map("a" -> 1.0, "b" -> 0.0), "a")
    ).toDS
    assertDatasetApproximateEquals(expected, predictions, 0.001)
  }

  "predict" should "predict and join labels for regression" in {
    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 1.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("4", MutableMap("feature1" -> 1.0, "feature2" -> 1.0)),
      FeatureSet("5", MutableMap("feature2" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("6", MutableMap("feature1" -> 1.0, "feature3" -> 1.0))
    ).toDS
    val labels = Seq(("1", 0.1), ("2", 0.2), ("3", 0.3)).toDS
    val model = new MockRegressionModel(v => DenseVector(v(0)), Map("feature1" -> 0, "feature2" -> 1, "feature3" -> 2))

    val predictions = Evaluator.predict(features, labels, model)(spark)
    val expected = Seq(
      (1.0, 0.1),
      (0.0, 0.2),
      (1.0, 0.3)
    ).toDS
    assertDatasetApproximateEquals(expected, predictions, 0.001)
  }

}
