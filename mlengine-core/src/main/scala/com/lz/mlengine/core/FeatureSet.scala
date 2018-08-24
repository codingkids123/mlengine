package com.lz.mlengine.core

import scala.collection.mutable.Map

case class FeatureSet(id: String, features: Map[String, Double] = Map())
