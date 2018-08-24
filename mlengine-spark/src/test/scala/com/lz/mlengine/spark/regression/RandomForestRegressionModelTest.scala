package com.lz.mlengine.spark.regression

import com.lz.mlengine.core.regression.RandomForestRegressionModel
import com.lz.mlengine.spark.{SparkConverter, SparkModelTest}
import org.apache.spark.ml.{regression => rg}
import org.junit.Test

class RandomForestRegressionModelTest extends SparkModelTest {

  @Test def testRegression() = {
    val sparkModel = getTrainer.fit(regressionData)
    val model = SparkConverter.convert(sparkModel)(Map[String, Int]())

    val path = s"${temporaryFolder.getRoot.getPath}/random_forest_regression"
    val modelLoaded = saveAndLoadModel(model, path, RandomForestRegressionModel.load)

    assertRegressionModelSame[rg.RandomForestRegressionModel](regressionData, sparkModel, modelLoaded)
  }

  def getTrainer = {
    new rg.RandomForestRegressor()
      .setMaxDepth(4)
  }

}
