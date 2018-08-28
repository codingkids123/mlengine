package com.lz.mlengine.spark.classification

import com.lz.mlengine.core.classification.GBTClassificationModel
import com.lz.mlengine.spark.{Converter, ModelTestBase}
import org.apache.spark.ml.{classification => cl}
import org.junit.Test

class GBTClassificationModelTestBase extends ModelTestBase {

  @Test def testBinaryClassification() = {
    val sparkModel = getTrainer.fit(binaryClassificationData)
    val model = Converter.convert(sparkModel)(Map[String, Int](), Map[Int, String]())

    val path = s"${temporaryFolder.getRoot.getPath}/gbt_classification_binary"
    val modelLoaded = saveAndLoadModel(model, path, GBTClassificationModel.load)

    assertBinaryClassificationModelProbabilitySame[cl.GBTClassificationModel](
      binaryClassificationData, sparkModel, modelLoaded
    )
  }

  def getTrainer = {
    new cl.GBTClassifier()
      .setMaxDepth(4)
  }

}
