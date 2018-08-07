package com.lz.mlengine

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification
import org.junit.{Rule, Test}
import org.junit.rules.TemporaryFolder
import org.scalatest.junit.JUnitSuite

import com.lz.mlengine.SparkConverter._

class LinearSVCModelTest extends JUnitSuite with DatasetSuiteBase {

  import spark.implicits._

  implicit val _ = spark

  val _temporaryFolder = new TemporaryFolder

  @Rule
  def temporaryFolder = _temporaryFolder

  @Test def testLinearSVC() = {
    val path = s"${temporaryFolder.getRoot.getPath}/linear_svc"
    val data = Seq(
      LabeledSparkFeature("1", Vectors.dense(Array(1.0, 1.0, 0.0)), 0.0),
      LabeledSparkFeature("2", Vectors.dense(Array(0.0, 1.0, 1.0)), 0.0),
      LabeledSparkFeature("3", Vectors.dense(Array(1.0, 0.0, 1.0)), 1.0)
    ).toDS

    val svc = new classification.LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)
    val sparkModel = svc.fit(data)
    SparkConverter.convert(sparkModel)(Map[String, Int](), Map[Int, String]()).save(path)
    val model = LinearSVCModel.load(path)

    val expected = sparkModel.transform(data).select("id", "rawPrediction").as[SparkPredictionRaw]
      .map(row => (row.id, row.rawPrediction(0), row.rawPrediction(1)))
    val predictions = data.map(row => new SparkPredictionRaw(row.id, model.predictImpl(row.features)))
      .map(row => (row.id, row.rawPrediction(0), row.rawPrediction(1)))

    assertDatasetApproximateEquals(expected, predictions, 0.001)
  }

}
