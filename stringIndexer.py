#!/usr/bin/env python
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import StringIndexerModel
spark = SparkSession.builder.appName('supervised_test').getOrCreate()
path = 'hdfs:/user/bm106/pub/project'
files = ['/cf_train.parquet','/cf_validation.parquet','/cf_test.parquet','/metadata.parquet','/features.parquet','/tags.parquet','/lyrics.parquet']
toy_file = './cf_toy.parquet'
data_files = ['./cf_train_subset.parquet','./cf_train_extra.parquet']
data_file = data_files[0]
#metadata_file = path+files[3]
u_idx_model_file = './sc2_final_u_indexer'
t_idx_model_file = './sc2_final_t_indexer'
data = spark.read.parquet(data_file)
#metadata = spark.read.parquet(metadata_file).select('track_id')
#uIDIndexer = StringIndexer(inputCol='user_id',outputCol='u_id')
#u_model = uIDIndexer.fit(data)
#u_model.save(u_idx_model_file)
u_model = StringIndexerModel.load(u_idx_model_file)
new_data = u_model.transform(data)
#tIDIndexer = StringIndexer(inputCol='track_id',outputCol='t_id')
#t_model = tIDIndexer.fit(metadata)
#t_model.save(t_idx_model_file)
t_model = StringIndexerModel.load(t_idx_model_file)
new_data = t_model.transform(new_data).select('u_id','t_id','count')

new_data_file = './cf_train_subset_idx_full.parquet'
new_data.write.parquet(new_data_file)
print('INDEXING FINISHED')
