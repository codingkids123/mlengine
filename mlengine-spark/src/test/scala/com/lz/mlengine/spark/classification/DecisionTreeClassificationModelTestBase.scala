package com.lz.mlengine.spark.classification

import com.lz.mlengine.core.classification.DecisionTreeClassificationModel
import com.lz.mlengine.spark.{Converter, ModelTestBase}
import org.apache.spark.ml.{classification => cl}
import org.junit.Test

class DecisionTreeClassificationModelTestBase extends ModelTestBase {

  @Test def testBinaryClassification() = {
    val sparkModel = getTrainer.fit(binaryClassificationData)
    val model = Converter.convert(sparkModel)(Map[String, Int](), Map[Int, String]())

    val path = s"${temporaryFolder.getRoot.getPath}/decision_tree_classification_binary"
    val modelLoaded = saveAndLoadModel(model, path, DecisionTreeClassificationModel.load)

    assertBinaryClassificationModelProbabilitySame[cl.DecisionTreeClassificationModel](
      binaryClassificationData, sparkModel, modelLoaded
    )
  }

  @Test def testMultiClassification() = {
    val sparkModel = getTrainer.fit(multiClassificationData)
    val model = Converter.convert(sparkModel)(Map[String, Int](), Map[Int, String]())

    val path = s"${temporaryFolder.getRoot.getPath}/decision_tree_classification_multiple"
    val modelLoaded = saveAndLoadModel(model, path, DecisionTreeClassificationModel.load)

    assertMultiClassificationModelProbabilitySame[cl.DecisionTreeClassificationModel](
      multiClassificationData, sparkModel, modelLoaded
    )
  }

  def getTrainer = {
    new cl.DecisionTreeClassifier()
      .setMaxDepth(4)
  }

}
