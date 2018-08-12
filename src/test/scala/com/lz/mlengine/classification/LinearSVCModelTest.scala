package com.lz.mlengine.classification

import com.lz.mlengine.{SparkConverter, SparkModelTest}
import org.apache.spark.ml.{classification => cl}
import org.junit.Test

class LinearSVCModelTest extends SparkModelTest {

  @Test def testBinaryClassification() = {
    val sparkModel = getTrainer.fit(binaryClassificationData)

    val path = s"${temporaryFolder.getRoot.getPath}/linear_svc"
    SparkConverter.convert(sparkModel)(Map[String, Int](), Map[Int, String]()).save(path)
    val model = LinearSVCModel.load(path)

    assertBinaryClassificationModelRawSame[cl.LinearSVCModel](binaryClassificationData, sparkModel, model)
  }

  def getTrainer() = {
    new cl.LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)
  }

}
