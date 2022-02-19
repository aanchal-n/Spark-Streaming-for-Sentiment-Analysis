from pyspark import SparkContext, SparkConf, sql
from pyspark.streaming import StreamingContext
import numpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron, SGDClassifier, PassiveAggressiveClassifier
import model
import preprocessing

TCP_PORT = 8000
batch_number = 1

def predict(record, spark, models): 
    record_schema = sql.types.StructType([
        sql.types.StructField('feature0', sql.types.StringType(), True),
        sql.types.StructField('feature1', sql.types.StringType(), True)
    ])
    if not record.isEmpty():
        df = spark.createDataFrame(record,schema=record_schema) # Converting DStream to a DataFrame

        labels = numpy.array([label[0] for label in numpy.array(df.select("feature0").collect())]) # Split dataset into labels and tweets
        tweets = numpy.array(df.select("feature1").collect())
        
        split_tweets = list(map(preprocessing.preprocess_tweet, tweets)) # Preprocess tweets and re-sentence
        tweet_sentences = list([' '.join(split_tweet) for split_tweet in split_tweets])
        
        batch_results = []
        for model in models:
            model.fit(tweet_sentences, labels, predict=True) # Fit model using a HashingVectorizer, and split dataset into training and validation
            batch_results.append([model.filename, model.get_metrics_on_test()])
        
        batch_results.sort(key=lambda classifier: classifier[1][0], reverse=True) # Log Results of all Classifiers
        global batch_number
        print(f"Batch #{batch_number}")
        for classifier in batch_results:
            print(f"{classifier[0]}: {classifier[1]}")
        batch_number += 1
        print("\n\n")

if __name__ == "__main__":
    conf = SparkConf().setAppName("Listening")
    sc = SparkContext(conf=conf)
    spark = sql.SparkSession(sc)
    ssc = StreamingContext(sc,1)
    #quiet_logs(sc)
    
    lines = ssc.socketTextStream('localhost', TCP_PORT)
    words = lines.flatMap(lambda word: preprocessing.format(word))
    classifiers_and_parameters = {
                                    MultinomialNB: {"alpha": 1}, 
                                    SGDClassifier: {"alpha":0.0001, "learning_rate":'optimal', "eta0":0.0, "power_t": 0.5},
                                    PassiveAggressiveClassifier: {"C":1.0}
                                  }
    
    models = [model.Model(classifier, classifier.__name__, classifiers_and_parameters[classifier], load_model=True) for classifier in classifiers_and_parameters]
    for model in models:
        model.load_classifier()
    words.foreachRDD(lambda rdd: predict(rdd, spark, models))
    
    ssc.start()
    ssc.awaitTermination() # This just waits infinitely; any way we can get it to stop after data is done streaming?
    
    
