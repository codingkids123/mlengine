package com.lz.mlengine.regression

import com.lz.mlengine.{SparkConverter, SparkModelTest}
import org.apache.spark.ml.{regression => rg}
import org.junit.Test

class RandomForestRegressionModelTest extends SparkModelTest {

  @Test def testRegression() = {
    val sparkModel = getTrainer.fit(regressionData)

    val path = s"${temporaryFolder.getRoot.getPath}/random_forest_regression"
    SparkConverter.convert(sparkModel)(Map[String, Int]()).save(path)
    val model = RandomForestRegressionModel.load(path)

    assertRegressionModelSame[rg.RandomForestRegressionModel](regressionData, sparkModel, model)
  }

  def getTrainer = {
    new rg.RandomForestRegressor()
      .setMaxDepth(4)
  }

}
