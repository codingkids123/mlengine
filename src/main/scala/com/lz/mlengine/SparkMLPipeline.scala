package com.lz.mlengine

import org.apache.spark.ml.classification._
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.types._

case class Label(id: String, label: String)

object SparkMLPipeline {

  // Classification.
  val DECISION_TREE = "DecisionTreeClassifier"
  val GBT_CLASSIFIER = "GBTClassifier"
  val LINEAR_SVC = "LinearSVC"
  val LOGISTIC_REGRESSION = "LogisticRegression"
  val MULTILAYER_PERCEPTRON = "MultilayerPerceptronClassifier"
  val NAIVE_BAYES = "NaiveBayes"
  val RANDOM_FOREST_CLASSIFIER = "RandomForestClassifier"

  // Regression.

  val MODE_TRAIN = "train"
  val MODE_PREDICT = "predict"

  def main(args: Array[String]): Unit = {
    if (args.length < 5) {
      println(
        "Wrong number of args! Correct args: <mode> <model type> <model path> <feature path> <label | prediction path>"
      )
      return
    }
    val mode = args(0)
    val modelType = args(1)
    val modelPath = args(2)
    val featurePath = args(3)
    val predictionPath = args(4)
    val labelPath = args(4)

    implicit val spark = SparkSession
      .builder()
      .appName("Spark ML Pipeline")
      .getOrCreate()

    mode match {
      case MODE_TRAIN => train(modelType, modelPath, featurePath, labelPath)
      case MODE_PREDICT => predict(modelType, modelPath, featurePath, predictionPath)
    }
  }

  def loadFeatures(path: String)(implicit spark: SparkSession): Dataset[FeatureSet] = {
    import spark.implicits._

    val schema = StructType(
      Seq(StructField("id", StringType), StructField("features", MapType(StringType, DoubleType)))
    )
    spark.read.schema(schema).json(path).as[FeatureSet]
  }

  def train(modelName: String, modelPath: String, featurePath: String, labelPath: String)
           (implicit spark: SparkSession): Unit = {
    import spark.implicits._

    val schema = StructType(Seq(StructField("id", StringType), StructField("label", StringType)))
    val labels = spark.read.schema(schema).csv(labelPath).as[Label]
      .map(l => PredictionSet(l.id, Seq(Prediction(Some(l.label), None))))

    val trainer = modelName match {
      case DECISION_TREE => {
        val sparkTrainer = new DecisionTreeClassifier()
        new SparkTrainer[DecisionTreeClassifier, DecisionTreeClassificationModel](sparkTrainer)
      }
      case GBT_CLASSIFIER => {
        val sparkTrainer = new GBTClassifier()
        new SparkTrainer[GBTClassifier, GBTClassificationModel](sparkTrainer)
      }
      case LINEAR_SVC => {
        val sparkTrainer = new LinearSVC()
        new SparkTrainer[LinearSVC, LinearSVCModel](sparkTrainer)
      }
      case LOGISTIC_REGRESSION => {
        val sparkTrainer = new LogisticRegression()
        new SparkTrainer[LogisticRegression, LogisticRegressionModel](sparkTrainer)
      }
      case MULTILAYER_PERCEPTRON => {
        val sparkTrainer = new MultilayerPerceptronClassifier()
        new SparkTrainer[MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel](sparkTrainer)
      }
      case NAIVE_BAYES => {
        val sparkTrainer = new NaiveBayes()
        new SparkTrainer[NaiveBayes, NaiveBayesModel](sparkTrainer)
      }
      case RANDOM_FOREST_CLASSIFIER => {
        val sparkTrainer = new RandomForestClassifier()
        new SparkTrainer[RandomForestClassifier, RandomForestClassificationModel](sparkTrainer)
      }
    }
    val features = loadFeatures(featurePath)
    trainer.fit(features, labels).save(modelPath)
  }

  def predict(modelName: String, modelPath: String, featurePath: String, predictionPath: String)
             (implicit spark: SparkSession): Unit = {
    val model = modelName match {
      case DECISION_TREE => SparkLoader.decisionTreeClassificationModel(modelPath)
      case GBT_CLASSIFIER => SparkLoader.gBTClassificationModel(modelPath)
      case LINEAR_SVC => SparkLoader.linearSVCModel(modelPath)
      case LOGISTIC_REGRESSION => SparkLoader.logisticRegressionModel(modelPath)
      case MULTILAYER_PERCEPTRON => SparkLoader.multilayerPerceptronClassificationModel(modelPath)
      case NAIVE_BAYES => SparkLoader.naiveBayesModel(modelPath)
      case RANDOM_FOREST_CLASSIFIER => SparkLoader.randomForestClassificationModel(modelPath)
    }
    val features = loadFeatures(featurePath)
    model.predict(features).write.format("json").save(predictionPath)
  }

}
