package com.lz.mlengine

import scala.collection.mutable.Map

import org.scalatest._

class FeatureSetTest extends FlatSpec with Matchers {

  "A FeatureSet" should "have a map of features" in {
    val featureSet = FeatureSet("1", Map("feature1" -> 1.0, "feature2" -> 2.0))
    featureSet.id should be ("1")
    featureSet.features.get("feature1").get should be (1.0)
    featureSet.features.get("feature2").get should be (2.0)
  }

}