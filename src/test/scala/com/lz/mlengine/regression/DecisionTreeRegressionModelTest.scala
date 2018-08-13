package com.lz.mlengine.regression

import com.lz.mlengine.{SparkConverter, SparkModelTest}
import org.apache.spark.ml.{regression => rg}
import org.junit.Test

class DecisionTreeRegressionModelTest extends SparkModelTest {

  @Test def testRegression() = {
    val sparkModel = getTrainer.fit(regressionData)
    val model = SparkConverter.convert(sparkModel)(Map[String, Int]())

    val path = s"${temporaryFolder.getRoot.getPath}/decision_tree_regression"
    val modelLoaded = saveAndLoadModel(model, path, DecisionTreeRegressionModel.load)

    assertRegressionModelSame[rg.DecisionTreeRegressionModel](regressionData, sparkModel, modelLoaded)
  }

  def getTrainer = {
    new rg.DecisionTreeRegressor()
      .setMaxDepth(4)
  }

}
