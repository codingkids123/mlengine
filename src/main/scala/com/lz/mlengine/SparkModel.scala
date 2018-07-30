package com.lz.mlengine

import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.{Dataset, SparkSession}

case class SparkFeature(id: String, features: Vector)

case class SparkPrediction(id: String, probability: Vector, rawPrediction: Vector, prediction: Double)

class SparkModel[M <: Model[M] with MLWritable](val model: M, val featureToIndexMap: Map[String, Int],
                                                val indexToLabelMap: Map[Int, String])
                                               (implicit spark: SparkSession) extends Serializable {
  import spark.implicits._

  def predict(features: Dataset[FeatureSet]): Dataset[PredictionSet] = {
    val sparkFeatures = getSparkFeatures(features)

    val sparkPredictions =
      model.transform(sparkFeatures).select("id", "probability", "rawPrediction", "prediction").as[SparkPrediction]

    getPredictionSets(sparkPredictions)
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

  private[mlengine] def getPredictionSets(sparkPredictions: Dataset[SparkPrediction]) = {
    sparkPredictions.map(row => {
      val predictions = Option(row.probability) match {
        case Some(probability) =>
          probability.toArray.zipWithIndex
            .map(p => Prediction(indexToLabelMap.get(p._2), Some(p._1))).toSeq
        case None =>
          Seq(Prediction(indexToLabelMap.get(row.prediction.toInt), None))
      }
      PredictionSet(row.id, predictions)
    })
  }

  def save(path: String): Unit = {
    model.save(s"${path}/model")
    featureToIndexMap.toSeq.toDS.write.format("parquet").save(s"${path}/feature_to_idx")
    indexToLabelMap.toSeq.toDS.write.format("parquet").save(s"${path}/idx_to_label")
  }

}
