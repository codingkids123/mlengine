package com.lz.mlengine

import org.apache.spark.ml.classification
import org.junit.Test

class LinearSVCModelTest extends SparkModelTest {

  @Test def testBinaryClassification() = {
    val sparkModel = getTrainer.fit(binaryClassificationData)

    val path = s"${temporaryFolder.getRoot.getPath}/linear_svc"
    SparkConverter.convert(sparkModel)(Map[String, Int](), Map[Int, String]()).save(path)
    val model = LinearSVCModel.load(path)

    assertBinaryClassificationModelRawSame[classification.LinearSVCModel](binaryClassificationData, sparkModel, model)
  }

  def getTrainer() = {
    new classification.LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)
  }

}
