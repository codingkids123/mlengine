package com.lz.mlengine.classification

import com.lz.mlengine.{SparkConverter, SparkModelTest}
import org.apache.spark.ml.{classification => cl}
import org.junit.Test

class LogisticRegressionModelTest extends SparkModelTest {

  @Test def testBinaryClassification() = {
    val sparkModel = getTrainer.fit(binaryClassificationData)

    val path = s"${temporaryFolder.getRoot.getPath}/logistic_regression_binary"
    SparkConverter.convert(sparkModel)(Map[String, Int](), Map[Int, String]()).save(path)
    val model = LogisticRegressionModel.load(path)

    assertBinaryClassificationModelProbabilitySame[cl.LogisticRegressionModel](
      binaryClassificationData, sparkModel, model
    )
  }

  @Test def testMultiClassification() = {
    val sparkModel = getTrainer.fit(multiClassificationData)

    val path = s"${temporaryFolder.getRoot.getPath}/logistic_regression_multiple"
    SparkConverter.convert(sparkModel)(Map[String, Int](), Map[Int, String]()).save(path)
    val model = LogisticRegressionModel.load(path)

    assertMultiClassificationModelProbabilitySame[cl.LogisticRegressionModel](
      multiClassificationData, sparkModel, model
    )
  }

  def getTrainer = {
    new cl.LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.1)
  }
}
