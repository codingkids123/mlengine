package com.lz.mlengine.spark

import com.lz.mlengine.core.{FeatureSet, MLModel, PredictionSet}
import org.apache.spark.sql.{Dataset, SparkSession}

class Metrics {

}


object Metrics {

  def evaluate(features: Dataset[FeatureSet], labels: Dataset[PredictionSet], model: MLModel)(implicit spark: SparkSession) = {
    import spark.implicits._

//    features.joinWith(labels, features.col("id") === labels.col("id"))
//      .map(row => {
//        val a = model.predict(row._1)
//      })

  }

}