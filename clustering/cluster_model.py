from sklearn.feature_extraction.text import HashingVectorizer
import pickle
import numpy
from sklearn.model_selection import train_test_split
from sklearn import metrics

k = 0

class Model:
    def __init__(self, classifier_type, filename, hyperparameters, load_model=False):
        self.vectorizer = HashingVectorizer(alternate_sign=False) # Vectorizer; alternate_sign is to prevent negatives
        self.vectors = None
        self.filename = filename
        self.batch_scores = []
        if not load_model:
            self.classifier = classifier_type(**hyperparameters)
            self.save_classifier() # Save blank classifier

    def load_classifier(self): # Load classifier from file for incremental learning
        with open(f"../models/{self.filename}.pkl", 'rb') as classifier_file:
            self.classifier = pickle.load(classifier_file)

    def save_classifier(self): # Save classifier after having updated the classifier with newest batch
        with open(f"../models/{self.filename}.pkl", 'wb') as classifier_file:
            pickle.dump(self.classifier, classifier_file)

    def fit(self, data, labels, predict=False): # Fit the data and split it
        self.labels = labels
        self.data = data
        self.vectors = self.vectorizer.fit_transform(data)
        if not predict:
            self.training_tweets, self.testing_tweets, self.training_labels, self.testing_labels = train_test_split(self.vectors, self.labels)
        
        else:
            self.testing_tweets, self.testing_labels = self.vectors, self.labels    
        self.testing_labels = [int(x) for x in self.testing_labels]


    def train(self, validation=False):
        self.load_classifier()
        
        if validation: # Train on validation set too, but only after checking accuracy on it.
            X = self.testing_tweets
        else:
            X = self.training_tweets
        
        self.classifier.partial_fit(X, y=None)

        if validation:
            self.save_classifier()
    
    def get_score(self): # Get Accuracy of model on validation set of batch, then train on validation too.
        output = self.classifier.predict(self.testing_tweets)
        """for i in range(len(output)):
            print(self.testing_labels[i], output[i])
        """
        predicted_labels = [0 if x > k else 4 for x in output]
        
        score = metrics.accuracy_score(self.testing_labels, predicted_labels)
        self.batch_scores.append(score)
        self.train(validation=True)
        return self.batch_scores[-1]
    
    def get_metrics_on_test(self):
        predicted_labels = self.classifier.predict(self.testing_tweets)
        
        predicted_labels = [1 if x == 4 else 0 for x in predicted_labels]
        self.testing_labels = [1 if x == 4 else 0 for x in self.testing_labels]

        score = metrics.accuracy_score(self.testing_labels, predicted_labels)
        confusion_matrix = metrics.confusion_matrix(self.testing_labels, predicted_labels)
        f1_score = metrics.f1_score(self.testing_labels, predicted_labels)
        recall = metrics.recall_score(self.testing_labels, predicted_labels)
        precision = metrics.precision_score(self.testing_labels, predicted_labels)

        return [score, confusion_matrix, f1_score, recall, precision]


    
