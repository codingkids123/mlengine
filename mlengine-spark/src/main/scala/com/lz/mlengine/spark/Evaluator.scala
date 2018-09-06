package com.lz.mlengine.spark


import com.lz.mlengine.core.{ClassificationMetrics, ClassificationModel, FeatureSet, RegressionMetrics, RegressionModel, Evaluator => CoreEvaluator}
import org.apache.spark.sql.{Dataset, SparkSession}

object Evaluator {

  type ClassificationLabel = Dataset[(String, String)]
  type RegressionLabel = Dataset[(String, Double)]

  def evaluate(features: Dataset[FeatureSet], labels: ClassificationLabel, model: ClassificationModel,
               numSteps: Int = 100)
              (implicit spark: SparkSession): ClassificationMetrics = {
    import spark.implicits._

    val predictions = features.joinWith(labels, features.col("id") === labels.col("_1"))
      .map { case (feature, label) => (model.predict(feature).predictions, label._2) }
      .collect

    CoreEvaluator.evaluate(predictions.toSeq, model.indexToLabelMap.values.toSeq, numSteps)
  }

  def evaluate(features: Dataset[FeatureSet], labels: RegressionLabel, model: RegressionModel)
              (implicit spark: SparkSession): RegressionMetrics = {
    import spark.implicits._

    val predictions = features.joinWith(labels, features.col("id") === labels.col("_1"))
      .map { case (feature, label) => (model.predict(feature).predictions.get("value").get, label._2) }
      .collect

    CoreEvaluator.evaluate(predictions.toSeq)
  }

}