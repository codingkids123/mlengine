package com.lz.mlengine.core

import java.io.{File, FileInputStream, FileOutputStream}

import scala.collection.mutable.{Map => MutableMap}

import breeze.linalg
import breeze.linalg.DenseVector
import org.junit.Assert._
import org.junit.rules.TemporaryFolder
import org.junit.{Rule, Test}
import org.scalatest.junit.JUnitSuite
import org.scalatest.Matchers

class MockClassificationModel(predictFun: (linalg.Vector[Double]) => linalg.Vector[Double],
                              override val featureToIndexMap: Map[String, Int],
                              override val indexToLabelMap: Map[Int, String])
  extends ClassificationModel(featureToIndexMap, indexToLabelMap) {

  override private[mlengine] def predictImpl(vector: linalg.Vector[Double]) = {
    predictFun(vector)
  }

}

object MockClassificationModel extends ModelLoader[MockClassificationModel]

class MockRegressionModel(predictFun: (linalg.Vector[Double]) => linalg.Vector[Double],
                          override val featureToIndexMap: Map[String, Int])
  extends RegressionModel(featureToIndexMap) {

  override private[mlengine] def predictImpl(vector: linalg.Vector[Double]) = {
    predictFun(vector)
  }

}

object MockRegressionModel extends ModelLoader[MockRegressionModel]

class ModelTest extends JUnitSuite with Matchers {

  val _temporaryFolder = new TemporaryFolder

  @Rule
  def temporaryFolder = _temporaryFolder

  @Test def testConvertFeatureSetToVector() = {
    val model = new MockRegressionModel(
      (_) => DenseVector(), Map[String, Int]("feature1" -> 0, "feature2" -> 1, "feature3" -> 2)
    )
    val feature = new FeatureSet("1", MutableMap("feature1" -> 1.0, "feature2" -> 2.0, "feature3" -> 3.0))
    val vector = model.convertFeatureSetToVector(feature)
    val expected = DenseVector(Array(1.0, 2.0, 3.0))
    assertArrayEquals(expected.toArray, vector.toDenseVector.toArray, 0.001)
  }

  @Test def testClassificationModelConvertVectorToPredictionSet() = {
    val model = new MockClassificationModel(
      (_) => DenseVector(), Map[String, Int](), Map[Int, String](0 -> "label0", 1 -> "label1")
    )
    val vector = DenseVector(Array(1.0, 2.0))
    val prediction = model.convertVectorToPredictionSet("1", vector)
    val expected = PredictionSet("1", Map("label0" -> 1.0, "label1" -> 2.0))
    assertEquals("1", prediction.id)
    assertEquals(expected.predictions.toSeq.sortBy(_._1).toString, prediction.predictions.toSeq.sortBy(_._1).toString)
  }

  @Test def testRegressionModelConvertVectorToPredictionSet() = {
    val model = new MockRegressionModel((_) => DenseVector(), Map[String, Int]())
    val vector = DenseVector(Array(1.0))
    val prediction = model.convertVectorToPredictionSet("1", vector)
    val expected = PredictionSet("1", Map("value" -> 1.0))
    assertEquals("1", prediction.id)
    assertEquals(expected.predictions.toSeq.sortBy(_._1).toString, prediction.predictions.toSeq.sortBy(_._1).toString)
  }

  @Test def testClassificationModelSaveAndLoadModel() = {
    val path = s"${temporaryFolder.getRoot.getPath}/model"
    val modelToSave = new MockClassificationModel(
      (_) => DenseVector(Array(1.0, 2.0)),
      Map[String, Int]("feature1" -> 0, "feature2" -> 1, "feature3" -> 2),
      Map[Int, String](0 -> "label0", 1 -> "label1")
    )

    val os = new FileOutputStream(new File(path))
    try {
      modelToSave.save(os)
    } finally {
      os.close
    }
    val is = new FileInputStream(new File(path))
    val modelLoaded = try {
      MockClassificationModel.load(is)
    } finally {
      is.close
    }

    assertEquals(
      modelToSave.featureToIndexMap.toSeq.sortBy(_._1).toString(),
      modelLoaded.featureToIndexMap.toSeq.sortBy(_._1).toString()
    )
    assertEquals(
      modelToSave.indexToLabelMap.toSeq.sortBy(_._1).toString(),
      modelLoaded.indexToLabelMap.toSeq.sortBy(_._1).toString()
    )
  }

  @Test def testRegressionModelSaveAndLoadModel() = {
    val path = s"${temporaryFolder.getRoot.getPath}/model"
    val modelToSave = new MockRegressionModel(
      (_) => DenseVector(Array(1.0, 2.0)),
      Map[String, Int]("feature1" -> 0, "feature2" -> 1, "feature3" -> 2)
    )

    val os = new FileOutputStream(new File(path))
    try {
      modelToSave.save(os)
    } finally {
      os.close
    }
    val is = new FileInputStream(new File(path))
    val modelLoaded = try {
      MockRegressionModel.load(is)
    } finally {
      is.close
    }

    assertEquals(
      modelToSave.featureToIndexMap.toSeq.sortBy(_._1).toString(),
      modelLoaded.featureToIndexMap.toSeq.sortBy(_._1).toString()
    )
  }

}
