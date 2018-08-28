package com.lz.mlengine.spark

import org.apache.spark.ml.{classification => cl}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{regression => rg}
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.{Dataset, SparkSession}

import com.lz.mlengine.core.{FeatureSet, MLModel, PredictionSet}

case class SparkFeature(id: String, feature: Vector)

case class LabeledSparkFeature(id: String, features: Vector, label: Double)

case class SparkPredictionProbability(id: String, probability: Vector)

case class SparkPredictionRaw(id: String, rawPrediction: Vector)

case class SparkPrediction(id: String, prediction: Double)

class Trainer[E <: Estimator[M], M <: Model[M] with MLWritable](val trainer: E)(implicit spark: SparkSession) {

  import spark.implicits._
  import Converter._

  def fit(features: Dataset[FeatureSet], labels: Option[Dataset[PredictionSet]]): MLModel = {
    implicit val featureToIndexMap = getFeatureToIndexMap(features)
    trainer match {
      case _: cl.DecisionTreeClassifier | _: cl.GBTClassifier | _: cl.LinearSVC |
           _: cl.LogisticRegression | _: cl.RandomForestClassifier => {
        val labelToIndexMap = getLabelToIndexMap(labels.get)
        val labeledVectors = getLabeledSparkFeature(features, labels.get, featureToIndexMap, Some(labelToIndexMap))
        implicit val indexToLabelMap = labelToIndexMap.map(_.swap)
        trainer match {
          case _: cl.DecisionTreeClassifier => {
            trainer.fit(labeledVectors).asInstanceOf[cl.DecisionTreeClassificationModel]
          }
          case _: cl.GBTClassifier => {
            trainer.fit(labeledVectors).asInstanceOf[cl.GBTClassificationModel]
          }
          case _: cl.LinearSVC => {
            trainer.fit(labeledVectors).asInstanceOf[cl.LinearSVCModel]
          }
          case _: cl.LogisticRegression => {
            trainer.fit(labeledVectors).asInstanceOf[cl.LogisticRegressionModel]
          }
          case _: cl.RandomForestClassifier => {
            trainer.fit(labeledVectors).asInstanceOf[cl.RandomForestClassificationModel]
          }
        }
      }
      case _: rg.DecisionTreeRegressor | _: rg.GBTRegressor | _: rg.LinearRegression | _: rg.RandomForestRegressor => {
        val labeledVectors = getLabeledSparkFeature(features, labels.get, featureToIndexMap, None)
        trainer match {
          case _: rg.DecisionTreeRegressor => {
            trainer.fit(labeledVectors).asInstanceOf[rg.DecisionTreeRegressionModel]
          }
          case _: rg.GBTRegressor => {
            trainer.fit(labeledVectors).asInstanceOf[rg.GBTRegressionModel]
          }
          case _: rg.LinearRegression => {
            trainer.fit(labeledVectors).asInstanceOf[rg.LinearRegressionModel]
          }
          case _: rg.RandomForestRegressor => {
            trainer.fit(labeledVectors).asInstanceOf[rg.RandomForestRegressionModel]
          }
        }
      }
      case _ => {
        throw new IllegalArgumentException(s"Unsupported model: ${trainer.getClass}")
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
