package com.lz.mlengine

import org.apache.spark.ml.regression
import org.junit.Test

class DecisionTreeRegressionModelTest extends SparkModelTest {

  @Test def testRegression() = {
    val sparkModel = getTrainer.fit(regressionData)

    val path = s"${temporaryFolder.getRoot.getPath}/decision_tree_regression"
    SparkConverter.convert(sparkModel)(Map[String, Int]()).save(path)
    val model = DecisionTreeRegressionModel.load(path)

    assertRegressionModelSame[regression.DecisionTreeRegressionModel](regressionData, sparkModel, model)
  }

  def getTrainer = {
    new regression.DecisionTreeRegressor()
      .setMaxDepth(4)
  }

}
