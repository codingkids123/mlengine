package com.lz.mlengine.regression

import com.lz.mlengine.{SparkConverter, SparkModelTest}
import org.apache.spark.ml.{regression => rg}
import org.junit.Test

class DecisionTreeRegressionModelTest extends SparkModelTest {

  @Test def testRegression() = {
    val sparkModel = getTrainer.fit(regressionData)

    val path = s"${temporaryFolder.getRoot.getPath}/decision_tree_regression"
    SparkConverter.convert(sparkModel)(Map[String, Int]()).save(path)
    val model = DecisionTreeRegressionModel.load(path)

    assertRegressionModelSame[rg.DecisionTreeRegressionModel](regressionData, sparkModel, model)
  }

  def getTrainer = {
    new rg.DecisionTreeRegressor()
      .setMaxDepth(4)
  }

}
