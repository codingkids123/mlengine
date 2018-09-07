package com.lz.mlengine.spark

import com.lz.mlengine.core.{ClassificationMetrics, ClassificationModel, FeatureSet, RegressionMetrics, RegressionModel, Evaluator => CoreEvaluator}
import org.apache.spark.sql.{Dataset, SparkSession}

object Evaluator {

  def evaluate(features: Dataset[FeatureSet], labels: Dataset[(String, String)], model: ClassificationModel,
               numSteps: Int = 100)
              (implicit spark: SparkSession): ClassificationMetrics = {
    CoreEvaluator.evaluate(predict(features, labels, model).collect.toSeq, model.indexToLabelMap.values.toSeq, numSteps)
  }

  def evaluate(features: Dataset[FeatureSet], labels: Dataset[(String, Double)], model: RegressionModel)
              (implicit spark: SparkSession): RegressionMetrics = {
    CoreEvaluator.evaluate(predict(features, labels, model).collect.toSeq)
  }

  private[mlengine] def predict(features: Dataset[FeatureSet], labels: Dataset[(String, String)],
                                model: ClassificationModel)
                               (implicit spark: SparkSession): Dataset[(Map[String, Double], String)] = {
    import spark.implicits._
    features.joinWith(labels, features.col("id") === labels.col("_1"))
      .map { case (feature, label) => (model.predict(feature).predictions, label._2) }
  }

  private[mlengine] def predict(features: Dataset[FeatureSet], labels: Dataset[(String, Double)],
                                model: RegressionModel)(implicit spark: SparkSession): Dataset[(Double, Double)] = {
    import spark.implicits._
    features.joinWith(labels, features.col("id") === labels.col("_1"))
      .map { case (feature, label) => (model.predict(feature).predictions.get("value").get, label._2) }
  }

}