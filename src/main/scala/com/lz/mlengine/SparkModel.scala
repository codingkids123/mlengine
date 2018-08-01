package com.lz.mlengine

import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

case class SparkFeature(id: String, features: Vector)

case class SparkPredictionVector(id: String, rawPrediction: Vector)

case class SparkPredictionScalar(id: String, prediction: Double)

class SparkModel[M <: Model[M] with MLWritable](val model: M, val featureToIndexMap: Map[String, Int],
                                                val indexToLabelMapMaybe: Option[Map[Int, String]])
                                               (implicit spark: SparkSession) extends Serializable {

  import spark.implicits._

  def predict(features: Dataset[FeatureSet]): Dataset[PredictionSet] = {
    getPredictionSets(model.transform(getSparkFeatures(features)))
  }

  private[mlengine] def getSparkFeatures(features: Dataset[FeatureSet]) = {
    features.map(row => {
      val values = row.features.toSeq
        .flatMap(kv => {
          featureToIndexMap.get(kv._1) match {
            case Some(index) => Seq((index, kv._2))
            case None => Seq()
          }
        })
        .sortBy(_._1)
      SparkFeature(row.id, Vectors.sparse(featureToIndexMap.size, values))
    })
  }

  private[mlengine] def getPredictionSets(sparkPredictions: DataFrame) = {
    indexToLabelMapMaybe match {
      case Some(_) =>
        sparkPredictions.select("id", "rawPrediction").as[SparkPredictionVector]
          .map(row => {
            val predictions = row.rawPrediction.toArray.zipWithIndex.map(p => {
              Prediction(indexToLabelMapMaybe.get.get(p._2), Some(p._1))
            })
            PredictionSet(row.id, predictions)
          })
      case None =>
        sparkPredictions.select("id", "prediction").as[SparkPredictionScalar]
          .map(row => {
            PredictionSet(row.id, Seq(Prediction(None, Some(row.prediction))))
          })
    }
  }

  def save(path: String): Unit = {
    model.save(s"${path}/model")
    featureToIndexMap.toSeq.toDS.write.format("parquet").save(s"${path}/feature_to_idx")
    indexToLabelMapMaybe match {
      case Some(indexToLabelMap) => {
        indexToLabelMap.toSeq.toDS.write.format("parquet").save(s"${path}/idx_to_label")
      }
      case None =>
    }
  }

}
