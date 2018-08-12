package com.lz.mlengine.regression

import com.lz.mlengine.{SparkConverter, SparkModelTest}
import org.apache.spark.ml.{regression => rg}
import org.junit.Test

class LinearRegressionModelTest extends SparkModelTest {

  @Test def testRegression() = {
    val sparkModel = getTrainer.fit(regressionData)

    val path = s"${temporaryFolder.getRoot.getPath}/linear_regression"
    SparkConverter.convert(sparkModel)(Map[String, Int]()).save(path)
    val model = LinearRegressionModel.load(path)

    assertRegressionModelSame[rg.LinearRegressionModel](regressionData, sparkModel, model)
  }

  def getTrainer = {
    new rg.LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.1)
  }

}
