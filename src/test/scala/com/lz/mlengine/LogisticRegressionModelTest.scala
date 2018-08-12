package com.lz.mlengine

import org.apache.spark.ml.classification
import org.junit.Test

class LogisticRegressionModelTest extends SparkModelTest {

  @Test def testBinaryClassification() = {
    val sparkModel = getTrainer.fit(binaryClassificationData)

    val path = s"${temporaryFolder.getRoot.getPath}/logistic_regression_binary"
    SparkConverter.convert(sparkModel)(Map[String, Int](), Map[Int, String]()).save(path)
    val model = LogisticRegressionModel.load(path)

    assertBinaryClassificationModelProbabilitySame[classification.LogisticRegressionModel](
      binaryClassificationData, sparkModel, model
    )
  }

  @Test def testMultiClassification() = {
    val sparkModel = getTrainer.fit(multiClassificationData)

    val path = s"${temporaryFolder.getRoot.getPath}/logistic_regression_multiple"
    SparkConverter.convert(sparkModel)(Map[String, Int](), Map[Int, String]()).save(path)
    val model = LogisticRegressionModel.load(path)

    assertMultiClassificationModelProbabilitySame[classification.LogisticRegressionModel](
      multiClassificationData, sparkModel, model
    )
  }

  def getTrainer = {
    new classification.LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.1)
  }
}
