package com.lz.mlengine

import scala.collection.mutable.{Map => MutableMap}
import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
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

  @Test def testSaveAndLoadModel() = {
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
    val modelToSave = trainer.fit(features, labels)

    val path = s"${temporaryFolder.getRoot.getPath}/test_model"
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
      modelToSave.indexToLabelMap.toSeq.sorted.toString,
      modelLoaded.indexToLabelMap.toSeq.sorted.toString
    )
  }

}
