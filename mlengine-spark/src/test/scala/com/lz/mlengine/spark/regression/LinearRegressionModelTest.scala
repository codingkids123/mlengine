package com.lz.mlengine.spark.regression

import com.lz.mlengine.core.regression.LinearRegressionModel
import com.lz.mlengine.spark.{SparkConverter, SparkModelTest}
import org.apache.spark.ml.{regression => rg}
import org.junit.Test

class LinearRegressionModelTest extends SparkModelTest {

  @Test def testRegression() = {
    val sparkModel = getTrainer.fit(regressionData)
    val model = SparkConverter.convert(sparkModel)(Map[String, Int]())

    val path = s"${temporaryFolder.getRoot.getPath}/linear_regression"
    val modelLoaded = saveAndLoadModel(model, path, LinearRegressionModel.load)

    assertRegressionModelSame[rg.LinearRegressionModel](regressionData, sparkModel, modelLoaded)
  }

  def getTrainer = {
    new rg.LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.1)
  }

}
