package com.lz.mlengine.spark

import java.net.URI

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.{classification => cl}
import org.apache.spark.ml.{regression => rg}
import org.apache.spark.sql.{Dataset, SparkSession}

import com.lz.mlengine.core.classification._
import com.lz.mlengine.core.regression._
import com.lz.mlengine.core.{FeatureSet, Model}

object Pipeline {

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

  val CLASSIFICATION_MODELS = Seq(DECISION_TREE_CLASSIFIER, GBT_CLASSIFIER, LINEAR_SVC, LOGISTIC_REGRESSION,
                                  RANDOM_FOREST_CLASSIFIER)
  val REGRESSION_MODELS = Seq(DECISION_TREE_REGRESSOR, GBT_REGRESSOR, LINEAR_REGRESSION, RANDOM_FOREST_REGRESSOR)
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
    spark.read.schema(Schema.featureSet).json(path).as[FeatureSet]
  }

  def loadClassificationLabels(path: String)(implicit spark: SparkSession): Dataset[(String, String)] = {
    import spark.implicits._
    spark.read.csv(path).map(l => (l.getString(0), l.getString(1)))
  }

  def loadRegressionLabels(path: String)(implicit spark: SparkSession): Dataset[(String, Double)] = {
    import spark.implicits._
    spark.read.csv(path).map(l => (l.getString(0), l.getString(1).toDouble))
  }

  def getTrainer(modelName: String)(implicit spark: SparkSession):Trainer[_, _] = {
    modelName match {
      // Classification models.
      case DECISION_TREE_CLASSIFIER =>
        new ClassificationTrainer[
          cl.DecisionTreeClassifier, cl.DecisionTreeClassificationModel](new cl.DecisionTreeClassifier())
      case GBT_CLASSIFIER =>
        new ClassificationTrainer[cl.GBTClassifier, cl.GBTClassificationModel](new cl.GBTClassifier())
      case LINEAR_SVC =>
        new ClassificationTrainer[cl.LinearSVC, cl.LinearSVCModel](new cl.LinearSVC())
      case LOGISTIC_REGRESSION =>
        new ClassificationTrainer[cl.LogisticRegression, cl.LogisticRegressionModel](new cl.LogisticRegression())
      case MULTILAYER_PERCEPTRON =>
        new ClassificationTrainer[cl.MultilayerPerceptronClassifier, cl.MultilayerPerceptronClassificationModel](
          new cl.MultilayerPerceptronClassifier())
      case NAIVE_BAYES =>
        new ClassificationTrainer[cl.NaiveBayes, cl.NaiveBayesModel](new cl.NaiveBayes())
      case RANDOM_FOREST_CLASSIFIER =>
        new ClassificationTrainer[
          cl.RandomForestClassifier, cl.RandomForestClassificationModel](new cl.RandomForestClassifier())

      // Regression models.
      case DECISION_TREE_REGRESSOR =>
        new RegressionTrainer[rg.DecisionTreeRegressor, rg.DecisionTreeRegressionModel](new rg.DecisionTreeRegressor())
      case GBT_REGRESSOR =>
        new RegressionTrainer[rg.GBTRegressor, rg.GBTRegressionModel](new rg.GBTRegressor())
      case GENERALIZED_LINEAR_REGRESSION =>
        new RegressionTrainer[rg.GeneralizedLinearRegression, rg.GeneralizedLinearRegressionModel](
          new rg.GeneralizedLinearRegression())
      case ISOTONIC_REGRESSION =>
        new RegressionTrainer[rg.IsotonicRegression, rg.IsotonicRegressionModel](new rg.IsotonicRegression())
      case LINEAR_REGRESSION =>
        new RegressionTrainer[rg.LinearRegression, rg.LinearRegressionModel](new rg.LinearRegression())
      case RANDOM_FOREST_REGRESSOR =>
        new RegressionTrainer[rg.RandomForestRegressor, rg.RandomForestRegressionModel](new rg.RandomForestRegressor())

      // Clustering models.
//      case K_MEANS =>
//        new Trainer[cs.KMeans, cs.KMeansModel](new cs.KMeans())
//      case GAUSSIAN_MIXTURE =>
//        new Trainer[cs.GaussianMixture, cs.GaussianMixtureModel](new cs.GaussianMixture())

      case _ => throw new IllegalArgumentException(s"Unsupported model: ${modelName}")
    }
  }

  def getModel(modelName: String, modelPath: String)(implicit spark: SparkSession): Model = {
    val fs = FileSystem.get(new URI(modelPath), spark.sparkContext.hadoopConfiguration)
    val file = new Path(modelPath)
    val fis = fs.open(file)
    try {
      modelName match {
        // TODO: Add more model support.
        // Classification models.
        case DECISION_TREE_CLASSIFIER => DecisionTreeClassificationModel.load(fis)
        case GBT_CLASSIFIER => GBTClassificationModel.load(fis)
        case LINEAR_SVC => LinearSVCModel.load(fis)
        case LOGISTIC_REGRESSION => LogisticRegressionModel.load(fis)
        case RANDOM_FOREST_CLASSIFIER => RandomForestClassificationModel.load(fis)

        // Regression models.
        case DECISION_TREE_REGRESSOR => DecisionTreeRegressionModel.load(fis)
        case GBT_REGRESSOR => GBTRegressionModel.load(fis)
        case LINEAR_REGRESSION => LinearRegressionModel.load(fis)
        case RANDOM_FOREST_REGRESSOR => RandomForestRegressionModel.load(fis)

        case _ => throw new IllegalArgumentException(s"Unsupported model: ${modelName}")
      }
    } finally {
      fis.close
    }
  }

  def train(modelName: String, modelPath: String, featurePath: String, labelPath: String)
           (implicit spark: SparkSession): Unit = {
    val trainer = getTrainer(modelName)
    val features = loadFeatures(featurePath)
    val model = if (CLASSIFICATION_MODELS.contains(modelName)) {
      val labels = loadClassificationLabels(labelPath)
      trainer.asInstanceOf[ClassificationTrainer[_, _]].fit(features, labels)
    } else if (REGRESSION_MODELS.contains(modelName)) {
      val labels = loadRegressionLabels(labelPath)
      trainer.asInstanceOf[RegressionTrainer[_, _]].fit(features, labels)
    } else {
      throw new IllegalArgumentException(s"Unsupported model: ${modelName}")
    }

    val fs = FileSystem.get(new URI(modelPath), spark.sparkContext.hadoopConfiguration)
    val file = new Path(modelPath)
    if (fs.exists(file)) fs.delete(file, false)
    val fos = fs.create(file)
    try {
      model.save(fos)
    } finally {
      fos.close
    }
  }

  def predict(modelName: String, modelPath: String, featurePath: String, predictionPath: String)
             (implicit spark: SparkSession): Unit = {
    import spark.implicits._
    val model = getModel(modelName, modelPath)
    val features = loadFeatures(featurePath)
    features.map(row => model.predict(row)).write.format("json").save(predictionPath)
  }

}
