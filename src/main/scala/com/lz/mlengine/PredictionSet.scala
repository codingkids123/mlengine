package com.lz.mlengine

case class Prediction(label: Option[String], value: Option[Double])

case class PredictionSet(id: String, predictions: Seq[Prediction])