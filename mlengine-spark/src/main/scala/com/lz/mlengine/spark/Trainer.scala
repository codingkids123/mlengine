package com.lz.mlengine.spark

import org.apache.spark.ml.{classification => cl}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{regression => rg}
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.{Dataset, SparkSession}

import com.lz.mlengine.core.{FeatureSet, MLModel}
import com.lz.mlengine.spark.Converter._

case class LabeledSparkFeature(id: String, features: Vector, label: Double)

abstract class Trainer[E <: Estimator[M], M <: Model[M] with MLWritable](val trainer: E)(implicit spark: SparkSession) {

  import spark.implicits._

  private[mlengine] def getFeatureToIndexMap(features: Dataset[FeatureSet]): Map[String, Int] = {
    features
      .flatMap(f => f.features.toSeq.map(_._1))
      .distinct()
      .collect()
      .sorted
      .zipWithIndex
      .toMap
  }

}

class ClassificationTrainer[E <: Estimator[M], M <: Model[M] with MLWritable]
  (override val trainer: E)(implicit spark: SparkSession) extends Trainer[E, M](trainer) {

  import spark.implicits._

  private[mlengine] def getLabelToIndexMap(labels: Dataset[(String, String)]): Map[String, Int] = {
    labels
      .map(_._2)
      .distinct()
      .collect()
      .sorted
      .zipWithIndex
      .toMap
  }

  private[mlengine] def getLabeledSparkFeature(features: Dataset[FeatureSet], labels: Dataset[(String, String)],
                                               featureToIndexMap: Map[String, Int],
                                               labelToIndexMap: Map[String, Int]): Dataset[LabeledSparkFeature] = {
    features
      .joinWith(labels, features.col("id") === labels.col("_1"))
      .map(row => {
        val values = row._1.features.toSeq.map(kv => (featureToIndexMap.get(kv._1).get, kv._2)).sortBy(_._1)
        val feature = Vectors.sparse(featureToIndexMap.size, values)
        val label = labelToIndexMap.get(row._2._2).get.toDouble
        LabeledSparkFeature(row._1.id, feature, label)
      })
  }

  def fit(features: Dataset[FeatureSet], labels: Dataset[(String, String)]): MLModel = {
    implicit val featureToIndexMap = getFeatureToIndexMap(features)
    val labelToIndexMap = getLabelToIndexMap(labels)
    val labeledVectors = getLabeledSparkFeature(features, labels, featureToIndexMap, labelToIndexMap)
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
      case _ => {
        throw new IllegalArgumentException(s"Unsupported model: ${trainer.getClass}")
      }
    }
  }

}

class RegressionTrainer[E <: Estimator[M], M <: Model[M] with MLWritable]
  (override val trainer: E)(implicit spark: SparkSession) extends Trainer[E, M](trainer) {

  import spark.implicits._

  private[mlengine] def getLabeledSparkFeature(features: Dataset[FeatureSet], labels: Dataset[(String, Double)],
                                               featureToIndexMap: Map[String, Int]): Dataset[LabeledSparkFeature] = {
    features
      .joinWith(labels, features.col("id") === labels.col("_1"))
      .map(row => {
        val values = row._1.features.toSeq.map(kv => (featureToIndexMap.get(kv._1).get, kv._2)).sortBy(_._1)
        val feature = Vectors.sparse(featureToIndexMap.size, values)
        val label = row._2._2
        LabeledSparkFeature(row._1.id, feature, label)
      })
  }

  def fit(features: Dataset[FeatureSet], labels: Dataset[(String, Double)]): MLModel = {
    implicit val featureToIndexMap = getFeatureToIndexMap(features)
    val labeledVectors = getLabeledSparkFeature(features, labels, featureToIndexMap)

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
      case _ => {
        throw new IllegalArgumentException(s"Unsupported model: ${trainer.getClass}")
      }
    }
  }

}