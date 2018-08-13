package com.lz.mlengine.classification

import com.lz.mlengine.{SparkConverter, SparkModelTest}
import org.apache.spark.ml.{classification => cl}
import org.junit.Test

class RandomForestClassificationModelTest extends SparkModelTest {

  @Test def testBinaryClassification() = {
    val sparkModel = getTrainer.fit(binaryClassificationData)

    val path = s"${temporaryFolder.getRoot.getPath}/random_forest_classification_binary"
    SparkConverter.convert(sparkModel)(Map[String, Int](), Map[Int, String]()).save(path)
    val model = RandomForestClassificationModel.load(path)

    assertBinaryClassificationModelProbabilitySame[cl.RandomForestClassificationModel](
      binaryClassificationData, sparkModel, model
    )
  }

  @Test def testMultiClassification() = {
    val sparkModel = getTrainer.fit(multiClassificationData)

    val path = s"${temporaryFolder.getRoot.getPath}/random_forest_classification_multiple"
    SparkConverter.convert(sparkModel)(Map[String, Int](), Map[Int, String]()).save(path)
    val model = RandomForestClassificationModel.load(path)

    assertMultiClassificationModelProbabilitySame[cl.RandomForestClassificationModel](
      multiClassificationData, sparkModel, model
    )
  }

  def getTrainer = {
    new cl.RandomForestClassifier()
      .setMaxDepth(4)
  }

}
