# MLEngine - A practical ML framework designed for production usecase

Build project
```
mvn install
```
Train a classification model
```
$SPARK_HOME/bin/spark-submit \
  --master local[4] \
  --class com.lz.mlengine.SparkMLPipeline target/mlengine-0.0.1.jar \
  train LogisticRegression \
  /tmp/lrmodel src/main/resources/sample_features.json src/test/resources/sample_classification_labels.txt
```
Use model to predict
```
$SPARK_HOME/bin/spark-submit \
  --master local[4] \
  --class com.lz.mlengine.SparkMLPipeline target/mlengine-0.0.1.jar \
  predict LogisticRegression \
  /tmp/lrmodel src/test/resources/sample_features.json /tmp/lrmodel/result
```

Train a regression model
```
$SPARK_HOME/bin/spark-submit \
  --master local[4] \
  --class com.lz.mlengine.SparkMLPipeline target/mlengine-0.0.1.jar \
  train LinearRegression \
  /tmp/lrmodel src/main/resources/sample_features.json src/test/resources/sample_regression_labels.txt
```
Use model to predict
```
$SPARK_HOME/bin/spark-submit \
  --master local[4] \
  --class com.lz.mlengine.SparkMLPipeline target/mlengine-0.0.1.jar \
  predict LinearRegression \
  /tmp/lrmodel src/test/resources/sample_features.json /tmp/lrmodel/result
```
