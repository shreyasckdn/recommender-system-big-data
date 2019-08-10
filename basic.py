#!/usr/bin/env python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('final_project').getOrCreate()

from datetime import datetime
from pyspark.mllib.recommendation import ALS,Rating
from pyspark.ml.feature import StringIndexer,StringIndexerModel
from pyspark.sql.functions import rank, col, size
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics

path = 'hdfs:/user/bm106/pub/project'
files = ['/cf_train.parquet','/cf_validation.parquet','/cf_test.parquet','/metadata.parquet','/features.parquet','/tags.parquet','/lyrics.parquet']
toy_file = './cf_toy.parquet'

data_files = ['./cf_train_subset_idx_full.parquet','./cf_train_subset_idx.parquet','./cf_train_idx.parquet','./cf_train_subset.parquet','./cf_train_extra.parquet']
model_file = './sc2_final1'

u_idx_model_file = './sc2_final_u_indexer'
t_idx_model_file = './sc2_final_t_indexer'

print(datetime.now())

data_file = path+files[1]

data = spark.read.parquet(data_file)
data = data.sample(False,0.001)
u_model = StringIndexerModel.load(u_idx_model_file)
t_model = StringIndexerModel.load(t_idx_model_file)

transformed_data = u_model.transform(data)
transformed_data = t_model.transform(transformed_data).select('u_id','t_id','count')

ratings = transformed_data.rdd.map(lambda l: Rating(l.u_id,l.t_id,l['count']))
rank = 10

model = ALS.trainImplicit(ratings, rank)

val_data_file = path+files[1]
val_data = spark.read.parquet(val_data_file)
val_data = val_data.sample(False,0.001)
transformed_val_data = u_model.transform(val_data)
transformed_val_data = t_model.transform(transformed_val_data).select('u_id','t_id','count')
#testdata = transformed_val_data.select('u_id','t_id')
#testdata = testdata.rdd.map(lambda l: (l[0],l[1]))
#predictions = model.predictAll(testdata)

labels = transformed_val_data.groupby('u_id').agg(F.collect_set('t_id').alias('ranked_labels'))
testusers = transformed_val_data.select('u_id').distinct().collect()
userSubsetRecs = model.recommendForUserSubset(testusers, 500)
#predictions = predictions.sort(col('count').desc(),col('t_id').desc())
#preds = predictions.groupby('u_id').agg(F.collect_set('rank').alias('ranked_predictions'))

joined_table = labels.join(preds,labels.u_id==preds.u_id)
reqdPredsLabels = joined_table.select(labels.ranked_labels,preds.ranked_predictions)
metrics = RankingMetrics(reqdPredsLabels)
print(metrics.precisionAt(500))
print(datetime.now())
