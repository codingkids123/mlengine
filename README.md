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
  --class com.lz.mlengine.SparkMLPipeline target/mlengine-0.0.1.jar \
  train LogisticRegression \
  /tmp/lrmodel src/test/resources/sample_features.json src/test/resources/sample_classification_labels.txt

# Predict
$SPARK_HOME/bin/spark-submit \
  --master local[4] \
  --class com.lz.mlengine.SparkMLPipeline target/mlengine-0.0.1.jar \
  predict LogisticRegression \
  /tmp/lrmodel src/test/resources/sample_features.json /tmp/lrmodel/result
```


Train a regression model
```
# Train
$SPARK_HOME/bin/spark-submit \
  --master local[4] \
  --class com.lz.mlengine.SparkMLPipeline target/mlengine-0.0.1.jar \
  train LinearRegression \
  /tmp/lrmodel src/test/resources/sample_features.json src/test/resources/sample_regression_labels.txt

# Predict
$SPARK_HOME/bin/spark-submit \
  --master local[4] \
  --class com.lz.mlengine.SparkMLPipeline target/mlengine-0.0.1.jar \
  predict LinearRegression \
  /tmp/lrmodel src/test/resources/sample_features.json /tmp/lrmodel/result
```

Train a clustering model
```
# Train
$SPARK_HOME/bin/spark-submit \
  --master local[4] \
  --class com.lz.mlengine.SparkMLPipeline target/mlengine-0.0.1.jar \
  train KMeans \
  /tmp/lrmodel src/test/resources/sample_features.json 

# Predict
$SPARK_HOME/bin/spark-submit \
  --master local[4] \
  --class com.lz.mlengine.SparkMLPipeline target/mlengine-0.0.1.jar \
  predict KMeans \
  /tmp/lrmodel src/test/resources/sample_features.json /tmp/lrmodel/result
```
