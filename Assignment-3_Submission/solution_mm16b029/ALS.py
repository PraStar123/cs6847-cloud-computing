from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row, SparkSession
import pydoop.hdfs as hdfs


def als_example(base_path, train_split, max_iters, reg_param):
	input_path = base_path + 'ALS.txt'
	try: 
		hdfs.get('/data_mm16b029/ALS.txt', input_path)
	except IOError:
		pass

	spark = SparkSession\
        .builder\
        .appName("ALSExample")\
        .getOrCreate()
	
	lines = spark.read.text(input_path).rdd
	parts = lines.map(lambda row: row.value.split("::"))
    
	ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2])))
	ratings = spark.createDataFrame(ratingsRDD)
	test_split = 1 - train_split
	(training, test) = ratings.randomSplit([train_split, test_split])

    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
	als = ALS(maxIter=max_iters, regParam=reg_param, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop")
	model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
	predictions = model.transform(test)
	evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
	rmse = evaluator.evaluate(predictions)
    #print("Root-mean-square error = " + str(rmse))

	return rmse 


