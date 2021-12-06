# import findspark
# findspark.init()
from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession, SQLContext, Row
from pyspark.sql.functions import col

from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, StringIndexer, NGram
from pyspark.ml import Pipeline, PipelineModel

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import NaiveBayes

import datetime


spark = SparkSession.builder.appName("SpamClassifier").getOrCreate()
#rw = spark.read.option("delimiter", "\t").csv("/home/pes1ug19cs341/Desktop/BigData/Project/train.csv").toDF('Subject', 'Message', 'Spam/ham')
#rw.show(5)
rw = spark.read.format("csv").option("header", "true").load("/home/pes1ug19cs341/Desktop/BigData/Project/train.csv")
rw.show(5)


# DATA PROCESSING
tknizer = Tokenizer().setInputCol('Message').setOutputCol('words')
stopwrds = StopWordsRemover().getStopWords() + ['-']
rem = StopWordsRemover().setStopWords(stopwrds).setInputCol('words').setOutputCol('filtered')
bigram = NGram().setN(2).setInputCol('filtered').setOutputCol('bigrams')

cvmodel = CountVectorizer().setInputCol('filtered').setOutputCol('features')
cvmodel_ngram = CountVectorizer().setInputCol('bigrams').setOutputCol('features')

ind = StringIndexer().setInputCol('Spam/Ham').setOutputCol('label')

pp = Pipeline(stages = [tknizer, rem, bigram, cvmodel, ind])
preprocessed = pp.fit(rw)
preprocessed.transform(rw).show(5)



mdlnb = NaiveBayes(smoothing = 1)
pipeline = Pipeline(stages = [tknizer, rem, cvmodel, ind, mdlnb])
mdl = pipeline.fit("train.csv")
predctn = mdl.transform("test.csv")
predctn.select('message', 'label', 'rawPrediction', 'probability', 'prediction').show(5)

eval = BinaryClassificationEvaluator().setLabelCol('label').setRawPredictionCol('prediction').setMetricName('areaUnderROC')
ac = eval.evaluate(predctn)
print("AUC: ", ac)

mdl.save('/home/pes1ug19cs341/Desktop/BigData/Project/')

if __name__ == "__main__":

    sc = SparkContext(appName = "SpamClassifier")
    ssc = StreamingContext(sc, 60)

    curr = datetime.datetime.now()
    fp = "/home/pes1ug19cs341/Desktop/BigData/Project/" + curr.strftime("%Y-%m-%d/")
    #print(fp)
    lines = ssc.textFileStream(fp)

    def process(t, rdd):
        if rdd.isEmpty():
            print("Input is Empty")
            return

        spark = SparkSession.builder.getOrCreate()
        rdd1 = rdd.map(lambda x : Row(message = x))
        df = spark.createDataFrame(rdd1)
        print(df.show())

        if not rdd.isEmpty():
            modl = PipelineModel.load('/home/pes1ug19cs341/Desktop/BigData/Project/mdl')

            pred = modl.transform(df)
            print(pred.show())

        

    lines.pprint()
    lines.foreachRDD(process)

    ssc.start()
    ssc.awaitTermination()














