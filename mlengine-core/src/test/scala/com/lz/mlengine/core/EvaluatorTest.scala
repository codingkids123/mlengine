package com.lz.mlengine.core

import org.scalactic.TolerantNumerics
import org.scalactic._
import org.scalatest._

class EvaluatorTest extends FlatSpec with Matchers {

  implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(0.001)

  implicit val classificationMetricsEq =
    new Equality[ClassificationMetrics] {
      def areEqual(a: ClassificationMetrics, b: Any): Boolean = {
        b match {
          case b: ClassificationMetrics =>
            a.confusionMatrices.toSeq.sortBy(_._1)
              .zip(b.confusionMatrices.toSeq.sortBy(_._1))
              .forall { case ((label1, cms1), (label2, cms2)) =>
                label1 == label2 &&
                  cms1.toSeq.sortBy(_._1).zip(cms2.toSeq.sortBy(_._1))
                    .forall { case ((threshold1, cm1), (threshold2, cm2)) => threshold1 === threshold2 && cm1 == cm2 }
              }
          case _ => false
        }
      }
    }

  implicit val regressionMetricsEq =
    new Equality[RegressionMetrics] {
      def areEqual(a: RegressionMetrics, b: Any): Boolean = {
        b match {
          case b: RegressionMetrics =>
            a.explainedVariance === b.explainedVariance && a.meanAbsoluteError === b.meanAbsoluteError &&
            a.meanSquaredError === b.meanSquaredError && a.r2 === b.r2
          case _ => false
        }
      }
    }

  "evaluate" should "calculate classification metrics" in {
    val predictions = Seq(
      (Map("a" -> 0.8, "b" -> 0.2), "a"),
      (Map("a" -> 0.5, "b" -> 0.5), "a"),
      (Map("a" -> 0.2, "b" -> 0.8), "b"),
      (Map("a" -> 0.6, "b" -> 0.4), "b")
    )
    val labels = Seq("a", "b")
    val metrics = Evaluator.evaluate(predictions, labels, 4)

    metrics should === (
      new ClassificationMetrics(Map(
        "a" -> Map(
          0.35 -> ConfusionMatrix(2, 1, 1, 0),
          0.5 -> ConfusionMatrix(1, 1, 1, 1),
          0.65 -> ConfusionMatrix(1, 0, 2, 1)),
        "b" -> Map(
          0.35 -> ConfusionMatrix(2, 1, 1, 0),
          0.5 -> ConfusionMatrix(1, 0, 2, 1),
          0.65 -> ConfusionMatrix(1, 0, 2, 1)
        )
      )))
  }

  "evaluate" should "calculate regression metrics" in {
    val predictions = Seq((0.2, 0.0), (0.5, 0.5), (0.8, 1.0))
    val metrics = Evaluator.evaluate(predictions)

    metrics should === (
      new RegressionMetrics(0.0, 0.0266667, 0.1333333, 0.84)
    )
  }

}
