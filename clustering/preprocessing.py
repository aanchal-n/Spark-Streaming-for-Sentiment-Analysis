import re
import nltk
from nltk.corpus import stopwords
import json

negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def preprocess_tweet(tweet):
    tweet = re.sub(r"(\s)@\w+", r"", tweet[0]) # Remove usernames
    tweet = re.sub(r"@\w+", r"", tweet) # Remove usernames that are at the beginning of a tweet
    
    tweet = re.sub(r"[0-9A-Za-z:./?=]+.com", r"", tweet) # Remove websites
    tweet = re.sub(r"[0-9A-Za-z:./?=]+.com\/[0-9A-Za-z:./?=]+", r"", tweet)
    tweet = re.sub(r"[0-9A-Za-z:./?=]+.net\/[0-9A-Za-z:./?=]+", r"", tweet)
    tweet = re.sub(r"[0-9A-Za-z:./?=]+.net", r"", tweet)
    tweet = re.sub(r"http[0-9A-Za-z:./?=]+", r"", tweet)    
    tweet = re.sub(r"[0-9]+\s", r"", tweet) # Remove numbers

    tweet = neg_pattern.sub(lambda x: negations_dic[x.group()], tweet)

    removals = r"!#,.$%&*()~`;=-+:" # Remove special characters
    for removal in removals:
        tweet = tweet.replace(removal, " ")

    tweet = re.sub(r"[0-9]+\s", r"", tweet) # Remove numbers again

    words = nltk.word_tokenize(tweet) # Tokenisation
    


    important_words = [word.lower() for word in words if word.lower() not in stopwords.words('english') or word.lower() == 'not'] # Stopword removal
    
    lemmatizer = nltk.stem.WordNetLemmatizer()

    return [lemmatizer.lemmatize(word, "a") for word in [lemmatizer.lemmatize(word) for word in important_words]]

def format(record): # Pre-processing function, changes weird json format to an array of data items
    parsed_data = json.loads(record)
    data_array = []
    for index in parsed_data:
        data_array.append(parsed_data[index])
    return data_array