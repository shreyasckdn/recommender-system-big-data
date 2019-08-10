#!/usr/bin/env python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('final_project').getOrCreate()

from datetime import datetime
from pyspark.ml.recommendation import ALS,ALSModel
from pyspark.ml.feature import StringIndexer,StringIndexerModel
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import explode

toy_file = './cf_toy.parquet'

print(datetime.now())
data = spark.read.parquet(toy_file)
uIDIndexer = StringIndexer(inputCol='user_id',outputCol='u_id')
tIDIndexer = StringIndexer(inputCol='track_id',outputCol='t_id')
new_data = uIDIndexer.fit(data).transform(data)
new_data = tIDIndexer.fit(new_data).transform(new_data)

als= ALS(maxIter=5, implicitPrefs=True, alpha=40, regParam=0, rank=20, userCol="u_id", itemCol="t_id", ratingCol="count", coldStartStrategy="drop")
model = als.fit(new_data)

labels = new_data.groupby('u_id').agg(F.collect_set('t_id').alias('ranked_labels'))
testusers = new_data.select('u_id').distinct()

userSubsetRecs = model.recommendForUserSubset(testusers, 10)
recommendationsDF = (userSubsetRecs.select("u_id", explode("recommendations").alias("recommendation")).select("u_id", "recommendation.*"))
preds = recommendationsDF.groupby('u_id').agg(F.collect_set('t_id').alias('ranked_preds'))
joined_table = labels.join(preds,labels.u_id==preds.u_id)
reqdPredsLabels = joined_table.select(labels.ranked_labels,preds.ranked_preds)
metrics = RankingMetrics(reqdPredsLabels.rdd)
print('Precision at 500 : {0}'.format(metrics.precisionAt(500)))
print('Mean Average Precision: {0}'.format(metrics.meanAveragePrecision))
print(datetime.now())


testdata = new_data.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = new_data.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))
print(datetime.now())

