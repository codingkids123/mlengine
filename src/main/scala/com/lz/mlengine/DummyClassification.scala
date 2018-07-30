package com.lz.mlengine

import scala.collection.mutable.Map
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.SparkSession

object DummyClassification {

  def main(args: Array[String]): Unit = {
    implicit val spark = SparkSession
      .builder()
      .appName("Dummy classification")
      .getOrCreate()

    import spark.implicits._

    println(s"Training dummy model ...")
    val lr = new LogisticRegression()
    lr.setMaxIter(10).setRegParam(0.01)

    val features = Seq(
      FeatureSet("1", Map("feature1" -> 1.0, "feature2" -> 0.0)),
      FeatureSet("2", Map("feature2" -> 1.0, "feature3" -> 0.0)),
      FeatureSet("3", Map("feature1" -> 1.0, "feature3" -> 0.0))
    ).toDS

    val labels = Seq(
      PredictionSet("1", Seq(Prediction(Some("positive"), None))),
      PredictionSet("2", Seq(Prediction(Some("negative"), None))),
      PredictionSet("3", Seq(Prediction(Some("negative"), None)))
    ).toDS

    val trainer = new SparkTrainer[LogisticRegression, LogisticRegressionModel](lr)

    val path = "/tmp/mlengine/dummy"
    println(s"Saving model to ${path}")
    trainer.fit(features, labels).save(path)

    println(s"Loading model from ${path}")
    val model = SparkLoader.logisticRegressionModel(path)

    println(s"Testing dummy model ...")
    val predictions = model.predict(features)

    println(s"Testing results:")
    predictions.collect().foreach {
      prediction => println(
        s"${prediction.id}: ${prediction.predictions.map( p => s"${p.label.get}: ${p.value.get}").mkString(", ")}"
      )
    }
  }

}