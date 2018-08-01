package com.lz.mlengine

import scala.collection.mutable.{Map => MutableMap}
import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.junit.Assert._
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.scalatest.junit.JUnitSuite

class SparkLoaderTest extends JUnitSuite with DatasetSuiteBase {

  import spark.implicits._

  implicit val _ = spark

  val _temporaryFolder = new TemporaryFolder

  @Rule
  def temporaryFolder = _temporaryFolder

  @Test def testSaveAndLoadClassificationModel() = {
    val lr = new LogisticRegression()
    lr.setMaxIter(10).setRegParam(0.01)

    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 0.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 0.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 0.0))
    ).toDS

    val labels = Seq(
      PredictionSet("1", Seq(Prediction(Some("positive"), None))),
      PredictionSet("2", Seq(Prediction(Some("negative"), None))),
      PredictionSet("3", Seq(Prediction(Some("negative"), None)))
    ).toDS

    val trainer = new SparkTrainer[LogisticRegression, LogisticRegressionModel](lr)
    val modelToSave = trainer.fit(features, Some(labels))

    val path = s"${temporaryFolder.getRoot.getPath}/classification_model"
    modelToSave.save(path)
    val modelLoaded = SparkLoader.logisticRegressionModel(path)

    assertEquals(modelToSave.model.uid, modelLoaded.model.uid)
    assertArrayEquals(modelToSave.model.coefficientMatrix.toArray, modelLoaded.model.coefficientMatrix.toArray, 0.001)
    assertArrayEquals(modelToSave.model.interceptVector.toArray, modelLoaded.model.interceptVector.toArray, 0.001)
    assertEquals(modelToSave.model.numClasses, modelLoaded.model.numClasses)
    assertEquals(
      modelToSave.featureToIndexMap.toSeq.sorted.toString,
      modelLoaded.featureToIndexMap.toSeq.sorted.toString
    )
    assertEquals(
      modelToSave.indexToLabelMapMaybe.get.toSeq.sorted.toString,
      modelLoaded.indexToLabelMapMaybe.get.toSeq.sorted.toString
    )
  }

  @Test def testSaveAndLoadRegressionModel() = {
    val lr = new LinearRegression()
    lr.setMaxIter(10).setRegParam(0.01)

    val features = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 0.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 0.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 0.0))
    ).toDS

    val labels = Seq(
      PredictionSet("1", Seq(Prediction(None, Some(0.8)))),
      PredictionSet("2", Seq(Prediction(None, Some(0.5)))),
      PredictionSet("3", Seq(Prediction(None, Some(0.2))))
    ).toDS

    val trainer = new SparkTrainer[LinearRegression, LinearRegressionModel](lr)
    val modelToSave = trainer.fit(features, Some(labels))

    val path = s"${temporaryFolder.getRoot.getPath}/regression_model"
    modelToSave.save(path)
    val modelLoaded = SparkLoader.linearRegressionModel(path)

    assertEquals(modelToSave.model.uid, modelLoaded.model.uid)
    assertArrayEquals(modelToSave.model.coefficients.toArray, modelLoaded.model.coefficients.toArray, 0.001)
    assertEquals(modelToSave.model.intercept, modelLoaded.model.intercept, 0.001)
    assertEquals(modelToSave.model.scale, modelLoaded.model.scale, 0.001)
    assertEquals(
      modelToSave.featureToIndexMap.toSeq.sorted.toString,
      modelLoaded.featureToIndexMap.toSeq.sorted.toString
    )
  }

}
