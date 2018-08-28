package com.lz.mlengine.spark.classification

import com.lz.mlengine.core.classification.LinearSVCModel
import com.lz.mlengine.spark.{Converter, ModelTestBase}
import org.apache.spark.ml.{classification => cl}
import org.junit.Test

class LinearSVCModelTestBase extends ModelTestBase {

  @Test def testBinaryClassification() = {
    val sparkModel = getTrainer.fit(binaryClassificationData)
    val model = Converter.convert(sparkModel)(Map[String, Int](), Map[Int, String]())

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
