package com.lz.mlengine

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.classification
import org.apache.spark.ml.linalg.Vectors
import org.junit.{Rule, Test}
import org.junit.rules.TemporaryFolder
import org.scalatest.junit.JUnitSuite
import com.lz.mlengine.SparkConverter._

class LogisticRegressionModelTest extends JUnitSuite with DatasetSuiteBase {

  import spark.implicits._

  implicit val _ = spark

  val _temporaryFolder = new TemporaryFolder

  @Rule
  def temporaryFolder = _temporaryFolder

  @Test def testBinaryClassification() = {
    val path = s"${temporaryFolder.getRoot.getPath}/logistic_regression_binary"
    val data = Seq(
      LabeledSparkFeature("1", Vectors.dense(Array(1.0, 1.0, 0.0)), 0.0),
      LabeledSparkFeature("2", Vectors.dense(Array(0.0, 1.0, 1.0)), 0.0),
      LabeledSparkFeature("3", Vectors.dense(Array(1.0, 0.0, 1.0)), 1.0)
    ).toDS

    val lr = new classification.LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.1)
    val sparkModel = lr.fit(data)
    SparkConverter.convert(sparkModel)(Map[String, Int](), Map[Int, String]()).save(path)
    val model = LogisticRegressionModel.load(path)

    val expected = sparkModel.transform(data).select("id", "probability").as[SparkPredictionProbability]
    val predictions = data.map(row => new SparkPredictionProbability(row.id, model.predictImpl(row.features)))

    assertDatasetApproximateEquals(expected, predictions, 0.001)
  }

  @Test def testMultinomialClassification() = {
    val path = s"${temporaryFolder.getRoot.getPath}/logistic_regression_multinomial"
    val data = Seq(
      LabeledSparkFeature("1", Vectors.dense(Array(1.0, 1.0, 0.0)), 0.0),
      LabeledSparkFeature("2", Vectors.dense(Array(0.0, 1.0, 1.0)), 0.0),
      LabeledSparkFeature("3", Vectors.dense(Array(1.0, 0.0, 1.0)), 1.0)
    ).toDS

    val lr = new classification.LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.1)
    val sparkModel = lr.fit(data)
    SparkConverter.convert(sparkModel)(Map[String, Int](), Map[Int, String]()).save(path)
    val model = LogisticRegressionModel.load(path)

    val expected = sparkModel.transform(data).select("id", "probability").as[SparkPredictionProbability]
    val predictions = data.map(row => new SparkPredictionProbability(row.id, model.predictImpl(row.features)))

    assertDatasetApproximateEquals(expected, predictions, 0.001)
  }
}
