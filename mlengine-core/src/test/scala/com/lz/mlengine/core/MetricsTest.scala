package com.lz.mlengine.core

import org.scalatest._

class ClassificationMetricsTest extends FlatSpec with Matchers {

  def getMetrics(label: String) = {
    new ClassificationMetrics(
      Map(label -> Map(
        0.0 -> ConfusionMatrix(4, 1, 9, 6),
        0.5 -> ConfusionMatrix(5, 3, 7, 5),
        1.0 -> ConfusionMatrix(8, 8, 2, 2)
      ))
    )
  }

  "prCurve" should "generate precision recall curve" in {
    getMetrics("test").prCurve("test") should be (Seq((0.4, 0.8), (0.5, 0.625), (0.8, 0.5)))
  }

  "rocCurve" should "generate roc curve" in {
    getMetrics("test").rocCurve("test") should be (Seq((0.0, 0.0), (0.1, 0.4), (0.3, 0.5), (0.8, 0.8), (1.0, 1.0)))
  }

  "areaUnderROC" should "calculate area under roc curve" in {
    getMetrics("test").areaUnderROC("test") should be (0.615)
  }

}

