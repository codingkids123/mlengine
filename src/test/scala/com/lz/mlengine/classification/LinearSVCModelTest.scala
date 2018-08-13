package com.lz.mlengine.classification

import com.lz.mlengine.{SparkConverter, SparkModelTest}
import org.apache.spark.ml.{classification => cl}
import org.junit.Test

class LinearSVCModelTest extends SparkModelTest {

  @Test def testBinaryClassification() = {
    val sparkModel = getTrainer.fit(binaryClassificationData)
    val model = SparkConverter.convert(sparkModel)(Map[String, Int](), Map[Int, String]())

    val path = s"${temporaryFolder.getRoot.getPath}/linear_svc"
    val modelLoaded = saveAndLoadModel(model, path, LinearSVCModel.load)

    assertBinaryClassificationModelRawSame[cl.LinearSVCModel](binaryClassificationData, sparkModel, modelLoaded)
  }

  def getTrainer() = {
    new cl.LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)
  }

}
