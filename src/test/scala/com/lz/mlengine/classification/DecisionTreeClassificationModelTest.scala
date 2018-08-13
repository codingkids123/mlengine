package com.lz.mlengine.classification

import com.lz.mlengine.{SparkConverter, SparkModelTest}
import org.apache.spark.ml.{classification => cl}
import org.junit.Test

class DecisionTreeClassificationModelTest extends SparkModelTest {

  @Test def testBinaryClassification() = {
    val sparkModel = getTrainer.fit(binaryClassificationData)
    val model = SparkConverter.convert(sparkModel)(Map[String, Int](), Map[Int, String]())

    val path = s"${temporaryFolder.getRoot.getPath}/decision_tree_classification_binary"
    val modelLoaded = saveAndLoadModel(model, path, DecisionTreeClassificationModel.load)

    assertBinaryClassificationModelProbabilitySame[cl.DecisionTreeClassificationModel](
      binaryClassificationData, sparkModel, modelLoaded
    )
  }

  @Test def testMultiClassification() = {
    val sparkModel = getTrainer.fit(multiClassificationData)
    val model = SparkConverter.convert(sparkModel)(Map[String, Int](), Map[Int, String]())

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
