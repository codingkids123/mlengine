package com.lz.mlengine

import org.scalatest._

class FeatureUtilTest extends FlatSpec with Matchers {

  "addCategoricalFeature" should "add categorical feature" in {
    val featureSet = FeatureSet("1")
    FeatureUtil.addCategoricalFeature("feature1", featureSet, "a")
    featureSet.features.get("feature1:a").get should be (1.0)
  }

  "addKeyValueFeature" should "add key value feature" in {
    val featureSet = FeatureSet("1")
    FeatureUtil.addKeyValueFeature("feature1", featureSet, Map("a" -> 0.0, "b" -> 1.0))
    featureSet.features.get("feature1:a").get should be (0.0)
    featureSet.features.get("feature1:b").get should be (1.0)
  }

  "addScalarFeature" should "add scalar feature" in {
    val featureSet = FeatureSet("1")
    FeatureUtil.addScalarFeature("feature1", featureSet, 1.0)
    featureSet.features.get("feature1").get should be (1.0)
  }

  "addVectorFeature" should "add vector feature" in {
    val featureSet = FeatureSet("1")
    FeatureUtil.addVectorFeature("feature1", featureSet, Seq(0.0, 1.0, 2.0))
    featureSet.features.get("feature1:0").get should be (0.0)
    featureSet.features.get("feature1:1").get should be (1.0)
    featureSet.features.get("feature1:2").get should be (2.0)
  }

  "addSparseFeature" should "add sparse feature" in {
    val featureSet = FeatureSet("1")
    FeatureUtil.addSparseFeature("feature1", featureSet, Map(0 -> 0.0, 2 -> 1.0))
    featureSet.features.get("feature1:0").get should be (0.0)
    featureSet.features.get("feature1:2").get should be (1.0)
  }

}