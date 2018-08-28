package com.lz.mlengine.spark

import scala.collection.mutable.{Map => MutableMap}
import com.holdenkarau.spark.testing.DatasetSuiteBase
import com.lz.mlengine.core.{FeatureSet, PredictionSet}
import org.junit.Assert._
import org.junit.{Rule, Test}
import org.junit.rules.TemporaryFolder
import org.scalatest.junit.JUnitSuite

class PipelineTest extends JUnitSuite with DatasetSuiteBase {

  import spark.implicits._

  implicit val _ = spark

  val _temporaryFolder = new TemporaryFolder

  @Rule
  def temporaryFolder = _temporaryFolder

  @Test def testLoadFeature = {
    val path = getClass.getClassLoader.getResource("sample_features.json").getFile()
    val features = Pipeline.loadFeatures(path)
    val expected = Seq(
      FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 1.0)),
      FeatureSet("2", MutableMap("feature2" -> 1.0, "feature3" -> 1.0)),
      FeatureSet("3", MutableMap("feature1" -> 1.0, "feature3" -> 1.0))
    ).toDS
    assertDatasetEquals(expected, features)
  }

  @Test def testTrainAndPredictClassification = {
    Pipeline.CLASSIFICATION_MODELS.foreach(model => {
      val featurePath = getClass.getClassLoader.getResource("sample_features.json").getFile()
      val labelPath = getClass.getClassLoader.getResource("sample_classification_labels.txt").getFile()
      val modelPath = s"${temporaryFolder.getRoot.getPath}/${model}/model"
      val predictionPath = s"${temporaryFolder.getRoot.getPath}/${model}/predictions"
      Pipeline.train(model, modelPath, featurePath, labelPath)
      Pipeline.predict(model, modelPath, featurePath, predictionPath)
      val predictions = spark.read.schema(Schema.predictionSet).json(predictionPath).as[PredictionSet].collect()
      assertEquals(3, predictions.length)
    })
  }

  @Test def testTrainAndPredictRegression = {
    Pipeline.REGRESSION_MODELS.foreach(model => {
      val featurePath = getClass.getClassLoader.getResource("sample_features.json").getFile()
      val labelPath = getClass.getClassLoader.getResource("sample_regression_labels.txt").getFile()
      val modelPath = s"${temporaryFolder.getRoot.getPath}/${model}/model"
      val predictionPath = s"${temporaryFolder.getRoot.getPath}/${model}/predictions"
      Pipeline.train(model, modelPath, featurePath, labelPath)
      Pipeline.predict(model, modelPath, featurePath, predictionPath)
      val predictions = spark.read.schema(Schema.predictionSet).json(predictionPath).as[PredictionSet].collect()
      assertEquals(3, predictions.length)
    })
  }

  @Test def testTrainAndPredictClustering = {
    Pipeline.CLUSTERING_MODELS.foreach(model => {
      val featurePath = getClass.getClassLoader.getResource("sample_features.json").getFile()
      val labelPath = ""
      val modelPath = s"${temporaryFolder.getRoot.getPath}/${model}/model"
      val predictionPath = s"${temporaryFolder.getRoot.getPath}/${model}/predictions"
      Pipeline.train(model, modelPath, featurePath, labelPath)
      Pipeline.predict(model, modelPath, featurePath, predictionPath)
      val predictions = spark.read.schema(Schema.predictionSet).json(predictionPath).as[PredictionSet].collect()
      assertEquals(3, predictions.length)
    })
  }

}
