package com.lz.mlengine

import org.apache.spark.ml.classification
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.{Dataset, SparkSession}

case class SparkFeature(id: String, feature: Vector)

case class LabeledSparkFeature(id: String, features: Vector, label: Double)

case class SparkPredictionProbability(id: String, probability: Vector)

case class SparkPredictionRaw(id: String, rawPrediction: Vector)

case class SparkPrediction(id: String, prediction: Double)

class SparkTrainer[E <: Estimator[M], M <: Model[M] with MLWritable](val trainer: E)(implicit spark: SparkSession) {

  import spark.implicits._
  import SparkConverter._

  def fit(features: Dataset[FeatureSet], labels: Option[Dataset[PredictionSet]]): MLModel = {
    implicit val featureToIndexMap = getFeatureToIndexMap(features)
    trainer match {
      case _: classification.LogisticRegression | _: classification.DecisionTreeClassifier => {
        val labelToIndexMap = getLabelToIndexMap(labels.get)
        val labeledVectors = getLabeledSparkFeature(features, labels.get, featureToIndexMap, Some(labelToIndexMap))
        implicit val indexToLabelMap = labelToIndexMap.map(_.swap)
        trainer match {
          case _: classification.LogisticRegression => {
            trainer.fit(labeledVectors).asInstanceOf[classification.LogisticRegressionModel]
          }
        }
      }
      case _: regression.LinearRegression => {
        val labeledVectors = getLabeledSparkFeature(features, labels.get, featureToIndexMap, None)
        trainer match {
          case _: regression.LinearRegression => {
            trainer.fit(labeledVectors).asInstanceOf[regression.LinearRegressionModel]
          }
        }
      }
    }
  }

  private[mlengine] def getFeatureToIndexMap(features: Dataset[FeatureSet]): Map[String, Int] = {
    features
      .flatMap(f => f.features.toSeq.map(_._1))
      .distinct()
      .collect()
      .sorted
      .zipWithIndex
      .toMap
  }

  private[mlengine] def getLabelToIndexMap(labels: Dataset[PredictionSet]): Map[String, Int] = {
    labels
      .flatMap(l => l.predictions.keys)
      .distinct()
      .collect()
      .sorted
      .zipWithIndex
      .toMap
  }

  private[mlengine] def getLabeledSparkFeature(features: Dataset[FeatureSet], labels: Dataset[PredictionSet],
                                               featureToIndexMap: Map[String, Int],
                                               labelToIndexMapMaybe: Option[Map[String, Int]]
                                              ): Dataset[LabeledSparkFeature] = {
    features
      .joinWith(labels, features.col("id") === labels.col("id"))
      .map(row => {
        val values = row._1.features.toSeq.map(kv => (featureToIndexMap.get(kv._1).get, kv._2)).sortBy(_._1)
        val feature = Vectors.sparse(featureToIndexMap.size, values)
        val label = labelToIndexMapMaybe match {
          case Some(labelToIndexMap) =>
            labelToIndexMap.get(row._2.predictions.head._1).get.toDouble
          case None =>
            row._2.predictions.get("value").get
        }
        LabeledSparkFeature(row._1.id, feature, label)
      })
  }

  private[mlengine] def getSparkFeature(features: Dataset[FeatureSet], featureToIndexMap: Map[String, Int]
                                       ): Dataset[SparkFeature] = {
    features.map(row => {
      val values = row.features.toSeq.map(kv => (featureToIndexMap.get(kv._1).get, kv._2)).sortBy(_._1)
      val feature = Vectors.sparse(featureToIndexMap.size, values)
      SparkFeature(row.id, feature)
    })
  }

}
