#!/usr/bin/env python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('final_project').getOrCreate()

from datetime import datetime
from pyspark.ml.recommendation import ALS,ALSModel
from pyspark.ml.feature import StringIndexer,StringIndexerModel
from pyspark.sql.functions import rank, log, col, size
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import explode

l=0
a=20
r=21

filter_data = spark.read.parquet('./cf_train_subset_idx_full_filter.parquet')
model_file = './models/filter_als_{0}_{1}_{2}.model'.format(l,a,r)
als=ALS(maxIter=5,implicitPrefs=True,alpha=a,regParam=l,rank=r,userCol="u_id",itemCol="t_id",ratingCol="count",coldStartStrategy="drop")
model = als.fit(filter_data)
model.save(model_file)

log_data = spark.read.parquet('./cf_train_subset_idx_full_log.parquet')
model_file = './models/log_als_{0}_{1}_{2}.model'.format(l,a,r)
als=ALS(maxIter=5,implicitPrefs=True,alpha=a,regParam=l,rank=r,userCol="u_id",itemCol="t_id",ratingCol="log_count",coldStartStrategy="drop")
model = als.fit(log_data)
model.save(model_file)

