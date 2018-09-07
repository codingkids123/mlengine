package com.lz.mlengine.core

import java.io._

import breeze.linalg.{Vector, VectorBuilder}

abstract class Model(val featureToIndexMap: Map[String, Int]) extends Serializable {

  def predict(feature: FeatureSet): PredictionSet = {
    val vector = convertFeatureSetToVector(feature)

    val prediction = predictImpl(vector)

    convertVectorToPredictionSet(feature.id, prediction)
  }

  def save(outputStream: OutputStream) = {
    val objectOutputStream = new ObjectOutputStream(outputStream)
    try {
      objectOutputStream.writeObject(this)
    } finally {
      objectOutputStream.close
    }
  }

  private[mlengine] def convertFeatureSetToVector(feature: FeatureSet): Vector[Double] = {
    val vb = new VectorBuilder[Double](featureToIndexMap.size)
    feature.features.toSeq
      .flatMap(kv => {
        featureToIndexMap.get(kv._1) match {
          case Some(index) => Seq((index, kv._2))
          case None => Seq()
        }
      })
      .sortBy(_._1)
      .foreach(item => vb.add(item._1, item._2))
    vb.toSparseVector
  }

  private[mlengine] def convertVectorToPredictionSet(id: String, vector: Vector[Double]): PredictionSet

  private[mlengine] def predictImpl(vector: Vector[Double]): Vector[Double]

}

abstract class ClassificationModel(override val featureToIndexMap: Map[String, Int],
                                   val indexToLabelMap: Map[Int, String]
                                  ) extends Model(featureToIndexMap) with Serializable {
  override private[mlengine] def convertVectorToPredictionSet(id: String, vector: Vector[Double]): PredictionSet = {
    new PredictionSet(
      id,
      vector.toArray.zipWithIndex.map(item => (indexToLabelMap.get(item._2).get, item._1)).toMap
    )
  }

}

abstract class RegressionModel(override val featureToIndexMap: Map[String, Int]) extends Model(featureToIndexMap)
  with Serializable {

  override private[mlengine] def convertVectorToPredictionSet(id: String, vector: Vector[Double]): PredictionSet = {
    new PredictionSet(id, Map("value" -> vector(0)))
  }

}

trait ModelLoader[M] {

  def load(inputStream: InputStream): M = {
    val objectInputStream = new ObjectInputStream(inputStream)
    try {
      return objectInputStream.readObject.asInstanceOf[M]
    } finally {
      objectInputStream.close()
    }
  }

}
