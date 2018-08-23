package com.lz.mlengine.regression

import com.lz.mlengine.{SparkConverter, SparkModelTest}
import org.apache.spark.ml.{regression => rg}
import org.junit.Test

class GBTRegressionModelTest extends SparkModelTest {

  @Test def testRegression() = {
    val sparkModel = getTrainer.fit(regressionData)
    val model = SparkConverter.convert(sparkModel)(Map[String, Int]())

    val path = s"${temporaryFolder.getRoot.getPath}/gbt_regression"
    val modelLoaded = saveAndLoadModel(model, path, GBTRegressionModel.load)

    assertRegressionModelSame[rg.GBTRegressionModel](regressionData, sparkModel, modelLoaded)
  }

  def getTrainer = {
    new rg.GBTRegressor()
      .setMaxDepth(4)
  }

}
