package com.lz.mlengine

import java.io.{ObjectInputStream, ObjectOutputStream}
import java.net.URI

import breeze.linalg.{Vector, VectorBuilder}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

abstract class MLModel(featureToIndexMap: Map[String, Int], indexToLabelMapMaybe: Option[Map[Int, String]]
                      ) extends Serializable {

  def predict(feature: FeatureSet): PredictionSet = {
    val vector = convertFeatureSetToVector(feature)

    val prediction = predictImpl(vector)

    convertVectorToPredictionSet(feature.id, prediction)
  }

  def save(path: String, overwrite: Boolean = false): Unit = {
    val configuration = new Configuration()
    val fs = FileSystem.get(new URI(path), configuration)
    val file = new Path(path)
    try {
      if (overwrite && fs.exists(file)) fs.delete(file, false)
      val fsdos = fs.create(file)
      val oos = new ObjectOutputStream(fsdos)
      try {
        oos.writeObject(this)
      } finally {
        oos.close
        fsdos.close
      }
    } finally {
      fs.close
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
  def load(path: String): M = {
    val configuration = new Configuration()
    val fs = FileSystem.get(new URI(path), configuration)
    val file = new Path(path)
    try {
      val fsdis = fs.open(file)
      val ois = new ObjectInputStream(fsdis)
      try {
        return ois.readObject.asInstanceOf[M]
      } finally {
        ois.close
        fsdis.close
      }
    } finally {
      fs.close
    }
  }
}
