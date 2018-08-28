package com.lz.mlengine.spark.classification

import com.lz.mlengine.core.classification.RandomForestClassificationModel
import com.lz.mlengine.spark.{Converter, ModelTestBase}
import org.apache.spark.ml.{classification => cl}
import org.junit.Test

class RandomForestClassificationModelTestBase extends ModelTestBase {

  @Test def testBinaryClassification() = {
    val sparkModel = getTrainer.fit(binaryClassificationData)
    val model = Converter.convert(sparkModel)(Map[String, Int](), Map[Int, String]())

    val path = s"${temporaryFolder.getRoot.getPath}/random_forest_classification_binary"
    val modelLoaded = saveAndLoadModel(model, path, RandomForestClassificationModel.load)

    assertBinaryClassificationModelProbabilitySame[cl.RandomForestClassificationModel](
      binaryClassificationData, sparkModel, modelLoaded
    )
  }

  @Test def testMultiClassification() = {
    val sparkModel = getTrainer.fit(multiClassificationData)
    val model = Converter.convert(sparkModel)(Map[String, Int](), Map[Int, String]())

    val path = s"${temporaryFolder.getRoot.getPath}/random_forest_classification_multiple"
    val modelLoaded = saveAndLoadModel(model, path, RandomForestClassificationModel.load)

    assertMultiClassificationModelProbabilitySame[cl.RandomForestClassificationModel](
      multiClassificationData, sparkModel, modelLoaded
    )
  }

  def getTrainer = {
    new cl.RandomForestClassifier()
      .setMaxDepth(4)
  }

}
