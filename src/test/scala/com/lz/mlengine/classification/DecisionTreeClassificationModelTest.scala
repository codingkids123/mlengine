package com.lz.mlengine.classification

import com.lz.mlengine.{SparkConverter, SparkModelTest}
import org.apache.spark.ml.{classification => cl}
import org.junit.Test

class DecisionTreeClassificationModelTest extends SparkModelTest {

  @Test def testBinaryClassification() = {
    val sparkModel = getTrainer.fit(binaryClassificationData)

    val path = s"${temporaryFolder.getRoot.getPath}/decision_tree_classification_binary"
    SparkConverter.convert(sparkModel)(Map[String, Int](), Map[Int, String]()).save(path)
    val model = DecisionTreeClassificationModel.load(path)

    assertBinaryClassificationModelProbabilitySame[cl.DecisionTreeClassificationModel](
      binaryClassificationData, sparkModel, model
    )
  }

  @Test def testMultiClassification() = {
    val sparkModel = getTrainer.fit(multiClassificationData)

    val path = s"${temporaryFolder.getRoot.getPath}/decision_tree_classification_multiple"
    SparkConverter.convert(sparkModel)(Map[String, Int](), Map[Int, String]()).save(path)
    val model = DecisionTreeClassificationModel.load(path)

    assertMultiClassificationModelProbabilitySame[cl.DecisionTreeClassificationModel](
      multiClassificationData, sparkModel, model
    )
  }

  def getTrainer = {
    new cl.DecisionTreeClassifier()
      .setMaxDepth(4)
  }

}
