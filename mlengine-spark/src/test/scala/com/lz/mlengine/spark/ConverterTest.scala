package com.lz.mlengine.spark

import org.scalatest._
import org.apache.spark.ml.linalg.{Matrices, Vectors}
import Converter._

class ConverterTest extends FlatSpec with Matchers {

  "convert" should "convert spark dense vector to dense vector and back" in {
    val vector = Vectors.dense(Array(1.0, 2.0, 3.0, 4.0))
    val vectorConverted = convert(convert(vector))
    vectorConverted should be (vector)
  }

  "convert" should "convert spark sparse vector to sparse vector and back" in {
    val vector = Vectors.sparse(4, Array(0, 2), Array(1.0, 2.0))
    val vectorConverted = convert(convert(vector))
    vectorConverted should be (vector)
  }

  "convert" should "convert spark dense matrix to dense matrix and back" in {
    val matrix = Matrices.dense(2, 3, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    val matrixConverted = convert(convert(matrix))
    matrixConverted should be (matrix)
  }

  "convert" should "convert spark sparse matrix to sparse matrix and back" in {
    val matrix = Matrices.sparse(3, 3, Array(0, 2, 3, 6), Array(0, 2, 1, 0, 1, 2), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    val matrixConverted = convert(convert(matrix))
    matrixConverted should be (matrix)
  }

}
