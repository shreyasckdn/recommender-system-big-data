#!/usr/bin/env python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('final_project').getOrCreate()
#.config("spark.sql.broadcastTimeout", "36000")
from datetime import datetime
from pyspark.ml.recommendation import ALS,ALSModel
from pyspark.ml.feature import StringIndexer,StringIndexerModel
from pyspark.sql.functions import rank, col, size
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import explode
import sys

path = 'hdfs:/user/bm106/pub/project'
files = ['/cf_train.parquet','/cf_validation.parquet','/cf_test.parquet','/metadata.parquet','/features.parquet','/tags.parquet','/lyrics.parquet']
toy_file = './cf_toy.parquet'

data_files = ['./cf_train_subset_idx_full.parquet','./cf_train_subset_idx.parquet','./cf_train_idx.parquet','./cf_train_subset.parquet','./cf_train_extra.parquet']
#print(datetime.now())
output = ""

val_files = ['./cf_val_idx_full.parquet','./cf_val_idx.parquet']
val_data_file = val_files[0]
val_data = spark.read.parquet(val_data_file)

labels = val_data.groupby('u_id').agg(F.collect_set('t_id').alias('ranked_labels'))
testusers = val_data.select('u_id').distinct()
model_files = ['models/als_0_1_13.model','models/als_0_1_21.model','models/als_0_1_5.model','models/als_0_20_13.model','models/als_0_20_21.model','models/als_0_20_5.model','models/als_0_40_13.model','models/als_0_40_21.model','models/als_0_40_5.model','models/als_100_1_13.model','models/als_100_1_21.model','models/als_100_1_5.model','models/als_100_20_13.model','models/als_100_20_21.model','models/als_100_20_5.model','models/als_100_40_13.model','models/als_100_40_21.model','models/als_100_40_5.model','models/als_50_1_13.model','models/als_50_1_21.model','models/als_50_1_5.model','models/als_50_20_13.model','models/als_50_20_21.model','models/als_50_20_5.model','models/als_50_40_13.model','models/als_50_40_21.model','models/als_50_40_5.model','models/filter_als_0_40_21.model','models/log_als_0_40_21.model']
c=1
for m in model_files:
    if c <= int(sys.argv[1]) :
        c += 1
        continue
    model_file = m
    #print('Working on {0}'.format(model_file))
    model = ALSModel.load(model_file)
    userSubsetRecs = model.recommendForUserSubset(testusers, 500)
    joined_table = labels.join(userSubsetRecs,labels.u_id==userSubsetRecs.u_id) 
    reqdPredsLabels = joined_table.rdd.map(lambda l: ([t.t_id for t in l.recommendations],l.ranked_labels))
    metrics = RankingMetrics(reqdPredsLabels)
    output += 'Results for model {0}\n'.format(model_file)
    output += 'Precision at 500 : {0}\n'.format(metrics.precisionAt(500))
    output += 'Mean Average Precision: {0}\n'.format(metrics.meanAveragePrecision)
    #print(datetime.now())
    c += 1
rdd = spark.sparkContext.parallelize([output])
rdd.saveAsTextFile('./metrics.txt')
