package com.lz.mlengine

import org.apache.spark.ml.classification
import org.apache.spark.ml.clustering
import org.apache.spark.ml.regression
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

  // Clustering.
  val K_MEANS = "KMeans"
  val GAUSSIAN_MIXTURE = "GaussianMixture"

  val CLASSIFICATION_MODELS = Seq(LOGISTIC_REGRESSION)
  val REGRESSION_MODELS = Seq(LINEAR_REGRESSION)
  val CLUSTERING_MODELS = Seq()

  val MODE_TRAIN = "train"
  val MODE_PREDICT = "predict"

  def main(args: Array[String]): Unit = {
    if (args.length < 4) {
      println(
        "Wrong number of args! Correct args: <mode> <model type> <model path> <feature path> [<label|prediction path>]"
      )
      return
    }
    val mode = args(0)
    val modelType = args(1)
    val modelPath = args(2)
    val featurePath = args(3)
    val predictionPath = if (args.length == 4) args(4) else ""
    val labelPath = if (args.length == 4) args(4) else ""

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
    spark.read.csv(path).map(l => PredictionSet(l.getString(0), Map(l.getString(1) -> 1.0)))
  }

  def loadRegressionLabels(path: String)(implicit spark: SparkSession): Dataset[PredictionSet] = {
    import spark.implicits._
    spark.read.csv(path).map(l => PredictionSet(l.getString(0), Map("value" -> l.getString(1).toDouble)))
  }

  def loadLabels(modelName: String, labelPath: String)(implicit spark: SparkSession): Option[Dataset[PredictionSet]] = {
    if (CLASSIFICATION_MODELS.contains(modelName)) {
      Some(loadClassificationLabels(labelPath))
    } else if (REGRESSION_MODELS.contains(modelName)) {
      Some(loadRegressionLabels(labelPath))
    } else if (CLUSTERING_MODELS.contains(modelName)) {
      None
    } else {
      throw new IllegalArgumentException(s"Unsupported model: ${modelName}")
    }
  }

  def getTrainer(modelName: String)(implicit spark: SparkSession):SparkTrainer[_, _] = {
    modelName match {
      // Classification models.
      case DECISION_TREE =>
        new SparkTrainer[classification.DecisionTreeClassifier, classification.DecisionTreeClassificationModel](
          new classification.DecisionTreeClassifier())
      case GBT_CLASSIFIER =>
        new SparkTrainer[classification.GBTClassifier, classification.GBTClassificationModel](
          new classification.GBTClassifier())
      case LINEAR_SVC =>
        new SparkTrainer[classification.LinearSVC, classification.LinearSVCModel](new classification.LinearSVC())
      case LOGISTIC_REGRESSION =>
        new SparkTrainer[classification.LogisticRegression, classification.LogisticRegressionModel](
          new classification.LogisticRegression())
      case MULTILAYER_PERCEPTRON =>
        new SparkTrainer[classification.MultilayerPerceptronClassifier,
                         classification.MultilayerPerceptronClassificationModel](
          new classification.MultilayerPerceptronClassifier())
      case NAIVE_BAYES =>
        new SparkTrainer[classification.NaiveBayes, classification.NaiveBayesModel](new classification.NaiveBayes())
      case RANDOM_FOREST_CLASSIFIER =>
        new SparkTrainer[classification.RandomForestClassifier, classification.RandomForestClassificationModel](
          new classification.RandomForestClassifier())

      // Regression models.
      case DECISION_TREE_REGRESSOR =>
        new SparkTrainer[regression.DecisionTreeRegressor, regression.DecisionTreeRegressionModel](
          new regression.DecisionTreeRegressor())
      case GBT_REGRESSOR =>
        new SparkTrainer[regression.GBTRegressor, regression.GBTRegressionModel](
          new regression.GBTRegressor())
      case GENERALIZED_LINEAR_REGRESSION =>
        new SparkTrainer[regression.GeneralizedLinearRegression, regression.GeneralizedLinearRegressionModel](
          new regression.GeneralizedLinearRegression())
      case ISOTONIC_REGRESSION =>
        new SparkTrainer[regression.IsotonicRegression, regression.IsotonicRegressionModel](
          new regression.IsotonicRegression())
      case LINEAR_REGRESSION =>
        new SparkTrainer[regression.LinearRegression, regression.LinearRegressionModel](
          new regression.LinearRegression())
      case RANDOM_FOREST_REGRESSOR =>
        new SparkTrainer[regression.RandomForestRegressor, regression.RandomForestRegressionModel](
          new regression.RandomForestRegressor())

      // Clustering models.
      case K_MEANS =>
        new SparkTrainer[clustering.KMeans, clustering.KMeansModel](new clustering.KMeans())
      case GAUSSIAN_MIXTURE =>
        new SparkTrainer[clustering.GaussianMixture, clustering.GaussianMixtureModel](new clustering.GaussianMixture())

      case _ => throw new IllegalArgumentException(s"Unsupported model: ${modelName}")
    }
  }

  def getModel(modelName: String, modelPath: String)(implicit spark: SparkSession):MLModel = {
    modelName match {
      // TODO: Add more model support.
      // Classification models.
      case LOGISTIC_REGRESSION => LogisticRegressionModel.load(modelPath)

      // Regression models.
      case LINEAR_REGRESSION => LinearRegressionModel.load(modelPath)

      case _ => throw new IllegalArgumentException(s"Unsupported model: ${modelName}")
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
    import spark.implicits._
    val model = getModel(modelName, modelPath)
    val features = loadFeatures(featurePath)
    features.map(row => model.predict(row)).write.format("json").save(predictionPath)
  }

}
