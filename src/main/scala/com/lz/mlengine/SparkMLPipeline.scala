package com.lz.mlengine

import org.apache.spark.ml.classification._
import org.apache.spark.ml.regression._
import org.apache.spark.sql.{Dataset, SparkSession}

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
  val DECISION_TREE_REGRESSOR = "DecisionTreeRegressor"
  val GBT_REGRESSOR = "GBTRegressor"
  val GENERALIZED_LINEAR_REGRESSION = "GeneralizedLinearRegression"
  val ISOTONIC_REGRESSION = "IsotonicRegression"
  val LINEAR_REGRESSION = "LinearRegression"
  val RANDOM_FOREST_REGRESSOR = "RandomForestRegressor"

  val CLASSIFICATION_MODELS = Seq(
    DECISION_TREE, GBT_CLASSIFIER, LINEAR_SVC, LOGISTIC_REGRESSION, NAIVE_BAYES, RANDOM_FOREST_CLASSIFIER
  )
  val REGRESSION_MODELS = Seq(
    DECISION_TREE_REGRESSOR, GBT_REGRESSOR, GENERALIZED_LINEAR_REGRESSION, ISOTONIC_REGRESSION,
    LINEAR_REGRESSION, RANDOM_FOREST_REGRESSOR
  )

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
    spark.read.schema(FeatureSet.schema).json(path).as[FeatureSet]
  }

  def loadClassificationLabels(path: String)(implicit spark: SparkSession): Dataset[PredictionSet] = {
    import spark.implicits._
    spark.read.csv(path).map(l => PredictionSet(l.getString(0), Seq(Prediction(Some(l.getString(1)), None))))
  }

  def loadRegressionLabels(path: String)(implicit spark: SparkSession): Dataset[PredictionSet] = {
    import spark.implicits._
    spark.read.csv(path).map(l => PredictionSet(l.getString(0), Seq(Prediction(None, Some(l.getString(1).toDouble)))))
  }

  def loadLabels(modelName: String, labelPath: String)(implicit spark: SparkSession): Option[Dataset[PredictionSet]] = {
    if (CLASSIFICATION_MODELS.contains(modelName)) {
      Some(loadClassificationLabels(labelPath))
    } else if (REGRESSION_MODELS.contains(modelName)) {
      Some(loadRegressionLabels(labelPath))
    } else {
      None
    }
  }

  def getTrainer(modelName: String)(implicit spark: SparkSession):SparkTrainer[_, _] = {
    modelName match {
      // Classification models.
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

      // Regression models.
      case DECISION_TREE_REGRESSOR => {
        val sparkTrainer = new DecisionTreeRegressor()
        new SparkTrainer[DecisionTreeRegressor, DecisionTreeRegressionModel](sparkTrainer)
      }
      case GBT_REGRESSOR => {
        val sparkTrainer = new GBTRegressor()
        new SparkTrainer[GBTRegressor, GBTRegressionModel](sparkTrainer)
      }
      case GENERALIZED_LINEAR_REGRESSION => {
        val sparkTrainer = new GeneralizedLinearRegression()
        new SparkTrainer[GeneralizedLinearRegression, GeneralizedLinearRegressionModel](sparkTrainer)
      }
      case ISOTONIC_REGRESSION => {
        val sparkTrainer = new IsotonicRegression()
        new SparkTrainer[IsotonicRegression, IsotonicRegressionModel](sparkTrainer)
      }
      case LINEAR_REGRESSION => {
        val sparkTrainer = new LinearRegression()
        new SparkTrainer[LinearRegression, LinearRegressionModel](sparkTrainer)
      }
      case RANDOM_FOREST_REGRESSOR => {
        val sparkTrainer = new RandomForestRegressor()
        new SparkTrainer[RandomForestRegressor, RandomForestRegressionModel](sparkTrainer)
      }
    }
  }

  def getModel(modelName: String, modelPath: String)(implicit spark: SparkSession):SparkModel[_] = {
    modelName match {
      // Classification models.
      case DECISION_TREE => SparkLoader.decisionTreeClassificationModel(modelPath)
      case GBT_CLASSIFIER => SparkLoader.gBTClassificationModel(modelPath)
      case LINEAR_SVC => SparkLoader.linearSVCModel(modelPath)
      case LOGISTIC_REGRESSION => SparkLoader.logisticRegressionModel(modelPath)
      case MULTILAYER_PERCEPTRON => SparkLoader.multilayerPerceptronClassificationModel(modelPath)
      case NAIVE_BAYES => SparkLoader.naiveBayesModel(modelPath)
      case RANDOM_FOREST_CLASSIFIER => SparkLoader.randomForestClassificationModel(modelPath)

      // Regression models.
      case DECISION_TREE_REGRESSOR => SparkLoader.decisionTreeRegressorModel(modelPath)
      case GBT_REGRESSOR => SparkLoader.gBTRegressionModel(modelPath)
      case GENERALIZED_LINEAR_REGRESSION => SparkLoader.generalizedLinearRegressionModel(modelPath)
      case ISOTONIC_REGRESSION => SparkLoader.isotonicRegressionModel(modelPath)
      case LINEAR_REGRESSION => SparkLoader.linearRegressionModel(modelPath)
      case RANDOM_FOREST_REGRESSOR => SparkLoader.randomForestRegressionModel(modelPath)
    }
  }

  def train(modelName: String, modelPath: String, featurePath: String, labelPath: String)
           (implicit spark: SparkSession): Unit = {
    val trainer = getTrainer(modelName)
    val features = loadFeatures(featurePath)
    val labels = loadLabels(modelName, labelPath)
    trainer.fit(features, labels).save(modelPath)
  }

  def predict(modelName: String, modelPath: String, featurePath: String, predictionPath: String)
             (implicit spark: SparkSession): Unit = {
    val model = getModel(modelName, modelPath)
    val features = loadFeatures(featurePath)
    model.predict(features).write.format("json").save(predictionPath)
  }

}
