package com.lz.mlengine

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression
import org.junit.{Rule, Test}
import org.junit.rules.TemporaryFolder
import org.scalatest.junit.JUnitSuite

import com.lz.mlengine.SparkConverter._

class LinearRegressionModelTest extends JUnitSuite with DatasetSuiteBase {

  import spark.implicits._

  implicit val _ = spark

  val _temporaryFolder = new TemporaryFolder

  @Rule
  def temporaryFolder = _temporaryFolder

  @Test def testLinearRegression() = {
    val path = s"${temporaryFolder.getRoot.getPath}/linear_regression"
    val data = Seq(
      LabeledSparkFeature("1", Vectors.dense(Array(1.0, 1.0, 0.0)), 0.0),
      LabeledSparkFeature("2", Vectors.dense(Array(0.0, 1.0, 1.0)), 0.5),
      LabeledSparkFeature("3", Vectors.dense(Array(1.0, 0.0, 1.0)), 1.0)
    ).toDS

    val lr = new regression.LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.1)
    val sparkModel = lr.fit(data)
    SparkConverter.convert(sparkModel)(Map[String, Int]()).save(path)
    val model = LinearRegressionModel.load(path)

    val expected = sparkModel.transform(data).select("id", "prediction").as[SparkPrediction]
    val predictions = data.map(row => new SparkPrediction(row.id, model.predictImpl(row.features)(0)))

    assertDatasetApproximateEquals(expected, predictions, 0.001)
  }

}
