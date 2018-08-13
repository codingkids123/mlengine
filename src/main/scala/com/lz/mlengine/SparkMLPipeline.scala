package com.lz.mlengine

import org.apache.spark.ml.{classification => cl}
import org.apache.spark.ml.{clustering => cs}
import org.apache.spark.ml.{regression => rg}
import org.apache.spark.sql.{Dataset, SparkSession}

object SparkMLPipeline {

  // Classification.
  val DECISION_TREE_CLASSIFIER = "DecisionTreeClassifier"
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

  val CLASSIFICATION_MODELS = Seq(DECISION_TREE_CLASSIFIER, LOGISTIC_REGRESSION, LINEAR_SVC, RANDOM_FOREST_CLASSIFIER)
  val REGRESSION_MODELS = Seq(DECISION_TREE_REGRESSOR, LINEAR_REGRESSION, RANDOM_FOREST_REGRESSOR)
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
    val predictionPath = if (args.length == 5) args(4) else ""
    val labelPath = if (args.length == 5) args(4) else ""

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
      case DECISION_TREE_CLASSIFIER =>
        new SparkTrainer[cl.DecisionTreeClassifier, cl.DecisionTreeClassificationModel](new cl.DecisionTreeClassifier())
      case GBT_CLASSIFIER =>
        new SparkTrainer[cl.GBTClassifier, cl.GBTClassificationModel](new cl.GBTClassifier())
      case LINEAR_SVC =>
        new SparkTrainer[cl.LinearSVC, cl.LinearSVCModel](new cl.LinearSVC())
      case LOGISTIC_REGRESSION =>
        new SparkTrainer[cl.LogisticRegression, cl.LogisticRegressionModel](new cl.LogisticRegression())
      case MULTILAYER_PERCEPTRON =>
        new SparkTrainer[cl.MultilayerPerceptronClassifier, cl.MultilayerPerceptronClassificationModel](
          new cl.MultilayerPerceptronClassifier())
      case NAIVE_BAYES =>
        new SparkTrainer[cl.NaiveBayes, cl.NaiveBayesModel](new cl.NaiveBayes())
      case RANDOM_FOREST_CLASSIFIER =>
        new SparkTrainer[cl.RandomForestClassifier, cl.RandomForestClassificationModel](new cl.RandomForestClassifier())

      // Regression models.
      case DECISION_TREE_REGRESSOR =>
        new SparkTrainer[rg.DecisionTreeRegressor, rg.DecisionTreeRegressionModel](new rg.DecisionTreeRegressor())
      case GBT_REGRESSOR =>
        new SparkTrainer[rg.GBTRegressor, rg.GBTRegressionModel](new rg.GBTRegressor())
      case GENERALIZED_LINEAR_REGRESSION =>
        new SparkTrainer[rg.GeneralizedLinearRegression, rg.GeneralizedLinearRegressionModel](
          new rg.GeneralizedLinearRegression())
      case ISOTONIC_REGRESSION =>
        new SparkTrainer[rg.IsotonicRegression, rg.IsotonicRegressionModel](new rg.IsotonicRegression())
      case LINEAR_REGRESSION =>
        new SparkTrainer[rg.LinearRegression, rg.LinearRegressionModel](new rg.LinearRegression())
      case RANDOM_FOREST_REGRESSOR =>
        new SparkTrainer[rg.RandomForestRegressor, rg.RandomForestRegressionModel](new rg.RandomForestRegressor())

      // Clustering models.
      case K_MEANS =>
        new SparkTrainer[cs.KMeans, cs.KMeansModel](new cs.KMeans())
      case GAUSSIAN_MIXTURE =>
        new SparkTrainer[cs.GaussianMixture, cs.GaussianMixtureModel](new cs.GaussianMixture())

      case _ => throw new IllegalArgumentException(s"Unsupported model: ${modelName}")
    }
  }

  def getModel(modelName: String, modelPath: String)(implicit spark: SparkSession):MLModel = {
    modelName match {
      // TODO: Add more model support.
      // Classification models.
      case DECISION_TREE_CLASSIFIER => classification.DecisionTreeClassificationModel.load(modelPath)
      case LINEAR_SVC => classification.LinearSVCModel.load(modelPath)
      case LOGISTIC_REGRESSION => classification.LogisticRegressionModel.load(modelPath)
      case RANDOM_FOREST_CLASSIFIER => classification.RandomForestClassificationModel.load(modelPath)

      // Regression models.
      case LINEAR_REGRESSION => regression.LinearRegressionModel.load(modelPath)
      case DECISION_TREE_REGRESSOR => regression.DecisionTreeRegressionModel.load(modelPath)
      case RANDOM_FOREST_REGRESSOR => regression.RandomForestRegressionModel.load(modelPath)

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
