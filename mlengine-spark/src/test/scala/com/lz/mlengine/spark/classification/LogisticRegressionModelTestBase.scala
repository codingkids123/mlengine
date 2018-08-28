package com.lz.mlengine.spark.classification

import com.lz.mlengine.core.classification.LogisticRegressionModel
import com.lz.mlengine.spark.{Converter, ModelTestBase}
import org.apache.spark.ml.{classification => cl}
import org.junit.Test

class LogisticRegressionModelTestBase extends ModelTestBase {

  @Test def testBinaryClassification() = {
    val sparkModel = getTrainer.fit(binaryClassificationData)
    val model = Converter.convert(sparkModel)(Map[String, Int](), Map[Int, String]())

    val path = s"${temporaryFolder.getRoot.getPath}/logistic_regression_binary"
    val modelLoaded = saveAndLoadModel(model, path, LogisticRegressionModel.load)

    assertBinaryClassificationModelProbabilitySame[cl.LogisticRegressionModel](
      binaryClassificationData, sparkModel, modelLoaded
    )
  }

  @Test def testMultiClassification() = {
    val sparkModel = getTrainer.fit(multiClassificationData)
    val model = Converter.convert(sparkModel)(Map[String, Int](), Map[Int, String]())

    val path = s"${temporaryFolder.getRoot.getPath}/logistic_regression_multiple"
    val modelLoaded = saveAndLoadModel(model, path, LogisticRegressionModel.load)

    assertMultiClassificationModelProbabilitySame[cl.LogisticRegressionModel](
      multiClassificationData, sparkModel, modelLoaded
    )
  }

  def getTrainer = {
    new cl.LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.1)
  }
}
