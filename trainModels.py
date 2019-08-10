#!/usr/bin/env python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('final_project').getOrCreate()

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

print('Hyper parameter training started...')

data_file = data_files[1]

data = spark.read.parquet(data_file)

reg_params = [0,50,100]
alphas = [1,20,40]
ranks = [5,13,21]
c=1
for l in reg_params:
    for a in alphas:
        for r in ranks:
            if c > int(sys.argv[1]):
                print(datetime.now())
                model_file = './models/als_{0}_{1}_{2}.model'.format(l,a,r)
                print('Working on {0}...'.format(model_file))
                als = ALS(maxIter=5, implicitPrefs = True, alpha = a,regParam=l, rank=r,userCol="u_id",itemCol="t_id",ratingCol="count",coldStartStrategy="drop")
                model = als.fit(data)
                model.save(model_file)
            c+=1
print(datetime.now())
print('Hyper parameter training finished')
