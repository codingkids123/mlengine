# MLEngine - A practical ML framework for production

Build project
```
git clone https://github.com/zhoulu312/mlengine.git
cd mlengine
mvn install
```

Train a classification model
```
# Train
$SPARK_HOME/bin/spark-submit \
  --master local[4] \
  --jars mlengine-core/target/mlengine-core-0.0.1.jar \
  --class com.lz.mlengine.spark.SparkMLPipeline mlengine-spark/target/mlengine-spark-0.0.1.jar \
  train LogisticRegression \
  /tmp/lrmodel mlengine-spark/src/test/resources/sample_features.json mlengine-spark/src/test/resources/sample_classification_labels.txt

# Predict
$SPARK_HOME/bin/spark-submit \
  --master local[4] \
  --jars mlengine-core/target/mlengine-core-0.0.1.jar \
  --class com.lz.mlengine.spark.SparkMLPipeline mlengine-spark/target/mlengine-spark-0.0.1.jar \
  predict LogisticRegression \
  /tmp/lrmodel mlengine-spark/src/test/resources/sample_features.json /tmp/lrmodel_result
```


Train a regression model
```
# Train
$SPARK_HOME/bin/spark-submit \
  --master local[4] \
  --jars mlengine-core/target/mlengine-core-0.0.1.jar \
  --class com.lz.mlengine.spark.SparkMLPipeline mlengine-spark/target/mlengine-spark-0.0.1.jar \
  train LinearRegression \
  /tmp/lrmodel mlengine-spark/src/test/resources/sample_features.json mlengine-spark/src/test/resources/sample_regression_labels.txt

# Predict
$SPARK_HOME/bin/spark-submit \
  --master local[4] \
  --jars mlengine-core/target/mlengine-core-0.0.1.jar \
  --class com.lz.mlengine.spark.SparkMLPipeline mlengine-spark/target/mlengine-spark-0.0.1.jar \
  predict LinearRegression \
  /tmp/lrmodel mlengine-spark/src/test/resources/sample_features.json /tmp/lrmodel_result
```
