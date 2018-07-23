package com.lz.mlengine

import org.scalatest._

class PredictionTest extends FlatSpec with Matchers {

  "A Prediction" should "have an option of label and an option of value" in {
    val prediction = Prediction("1", Some("positive"), Some(0.8))
    prediction.id should be ("1")
    prediction.label.get should be ("positive")
    prediction.value.get should be (0.8)
  }

}