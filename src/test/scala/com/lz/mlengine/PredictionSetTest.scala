package com.lz.mlengine

import org.scalatest._

class PredictionTest extends FlatSpec with Matchers {

  "A Prediction" should "have an option of label and an option of value" in {
    val predictionSet = PredictionSet(
      "1",
      Seq(Prediction(Some("positive"), Some(0.8)), Prediction(Some("negative"), Some(0.2)))
    )
    predictionSet.id should be ("1")
    predictionSet.predictions(0).label.get should be ("positive")
    predictionSet.predictions(0).value.get should be (0.8)
    predictionSet.predictions(1).label.get should be ("negative")
    predictionSet.predictions(1).value.get should be (0.2)
  }

}