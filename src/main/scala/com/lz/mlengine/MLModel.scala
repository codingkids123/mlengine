package com.lz.mlengine

import java.io._

import breeze.linalg.{Vector, VectorBuilder}

abstract class MLModel(featureToIndexMap: Map[String, Int], indexToLabelMapMaybe: Option[Map[Int, String]]
                      ) extends Serializable {

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

  private[mlengine] def convertVectorToPredictionSet(id: String, vector: Vector[Double]): PredictionSet = {
    indexToLabelMapMaybe match {
      case Some(indexToLabelMap) =>
        new PredictionSet(
          id,
          vector.toArray.zipWithIndex.map(item => (indexToLabelMap.get(item._2).get, item._1)).toMap
        )
      case None => new PredictionSet(id, Map("value" -> vector(0)))
    }
  }

  private[mlengine] def predictImpl(vector: Vector[Double]): Vector[Double]

}

trait MLModelLoader[M] {

  def load(inputStream: InputStream): M = {
    val objectInputStream = new ObjectInputStream(inputStream)
    try {
      return objectInputStream.readObject.asInstanceOf[M]
    } finally {
      objectInputStream.close()
    }
  }

}
