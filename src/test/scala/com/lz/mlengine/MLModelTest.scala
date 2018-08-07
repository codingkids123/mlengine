package com.lz.mlengine

import scala.collection.mutable.{Map => MutableMap}

import breeze.linalg.DenseVector
import breeze.linalg.Vector
import org.junit.Assert._
import org.junit.rules.TemporaryFolder
import org.junit.{Rule, Test}
import org.scalatest.junit.JUnitSuite
import org.scalatest.Matchers

class MockModel(val prediction: Vector[Double], val featureToIndexMap: Map[String, Int],
                val indexToLabelMapMaybe: Option[Map[Int, String]])
  extends MLModel(featureToIndexMap, indexToLabelMapMaybe) {

  override private[mlengine] def predictImpl(vector: Vector[Double]): Vector[Double] = prediction

}

object MockModel extends MLModelLoader[MockModel]

class MLModelTest extends JUnitSuite with Matchers {

  val _temporaryFolder = new TemporaryFolder

  @Rule
  def temporaryFolder = _temporaryFolder

  @Test def testConvertFeatureSetToVector() = {
    val model = new MockModel(DenseVector(), Map[String, Int]("feature1" -> 0, "feature2" -> 1, "feature3" -> 2), None)
    val feature = new FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 2.0, "feature3" -> 3.0))
    val vector = model.convertFeatureSetToVector(feature)
    val expected = DenseVector(Array(1.0, 2.0, 3.0))
    assertArrayEquals(expected.toArray, vector.toDenseVector.toArray, 0.001)
  }

  @Test def convertVectorToPredictionSetWithIndexToLabelMap() = {
    val model = new MockModel(DenseVector(), Map[String, Int](), Some(Map[Int, String](0 -> "label0", 1 -> "label1")))
    val vector = DenseVector(Array(1.0, 2.0))
    val prediction = model.convertVectorToPredictionSet("1", vector)
    val expected = PredictionSet("1", Map("label0" -> 1.0, "label1" -> 2.0))
    assertEquals("1", prediction.id)
    assertEquals(expected.predictions.toSeq.sortBy(_._1).toString, prediction.predictions.toSeq.sortBy(_._1).toString)
  }

  @Test def convertVectorToPredictionSetWithoutIndexToLabelMap() = {
    val model = new MockModel(DenseVector(), Map[String, Int](), None)
    val vector = DenseVector(Array(1.0))
    val prediction = model.convertVectorToPredictionSet("1", vector)
    val expected = PredictionSet("1", Map("value" -> 1.0))
    assertEquals("1", prediction.id)
    assertEquals(expected.predictions.toSeq.sortBy(_._1).toString, prediction.predictions.toSeq.sortBy(_._1).toString)
  }

  @Test def testSaveAndLoadModelWithIndexToLabelMap() = {
    val path = s"${temporaryFolder.getRoot.getPath}/model"
    val modelToSave = new MockModel(
      DenseVector(Array(1.0, 2.0)),
      Map[String, Int]("feature1" -> 0, "feature2" -> 1, "feature3" -> 2),
      Some(Map[Int, String](0 -> "label0", 1 -> "label1"))
    )
    modelToSave.save(path)
    val modelLoaded = MockModel.load(path)
    assertEquals(
      modelToSave.featureToIndexMap.toSeq.sortBy(_._1).toString(),
      modelLoaded.featureToIndexMap.toSeq.sortBy(_._1).toString()
    )
    assertEquals(
      modelToSave.indexToLabelMapMaybe.get.toSeq.sortBy(_._1).toString(),
      modelLoaded.indexToLabelMapMaybe.get.toSeq.sortBy(_._1).toString()
    )
  }

  @Test def testSaveAndLoadModelWithoutIndexToLabelMap() = {
    val path = s"${temporaryFolder.getRoot.getPath}/model"
    val modelToSave = new MockModel(
      DenseVector(Array(1.0, 2.0)),
      Map[String, Int]("feature1" -> 0, "feature2" -> 1, "feature3" -> 2),
      None
    )
    modelToSave.save(path)
    val modelLoaded = MockModel.load(path)
    assertEquals(
      modelToSave.featureToIndexMap.toSeq.sortBy(_._1).toString(),
      modelLoaded.featureToIndexMap.toSeq.sortBy(_._1).toString()
    )
    assertTrue(modelLoaded.indexToLabelMapMaybe.isEmpty)
  }

}
