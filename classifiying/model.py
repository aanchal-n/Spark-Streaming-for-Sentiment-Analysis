from sklearn.feature_extraction.text import HashingVectorizer
import pickle
import numpy
from sklearn.model_selection import train_test_split
from sklearn import metrics

class Model:
    def __init__(self, classifier_type, filename, hyperparameters, load_model=False):
        self.vectorizer = HashingVectorizer(n_features=2**25, alternate_sign=False) # Vectorizer; alternate_sign is to prevent negatives
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
        self.classifier = None

    def fit(self, data, labels, predict=False): # Fit the data and split it
        self.labels = labels
        self.vectors = self.vectorizer.fit_transform(data)
        if not predict:
            self.training_tweets, self.testing_tweets, self.training_labels, self.testing_labels = train_test_split(self.vectors, self.labels)
        else:
            self.testing_tweets, self.testing_labels = self.vectors, self.labels    

    def train(self, validation=False):
        self.load_classifier()
        
        if validation: # Train on validation set too, but only after checking accuracy on it.
            X = self.testing_tweets
            y = self.testing_labels
        else:
            X = self.training_tweets
            y = self.training_labels
        
        self.classifier.partial_fit(X, y, classes=numpy.unique(y)) # Partial fit is the key to incremental learning here; It updates weights with new batch as opposed to overwriting them

        if validation:
            self.save_classifier()
    
    def get_score(self): # Get Accuracy of model on validation set of batch, then train on validation too.
        self.batch_scores.append(self.classifier.score(self.testing_tweets, self.testing_labels))
        self.train(validation=True,)
        return self.batch_scores[-1]
    
    def get_metrics_on_test(self):
        predicted_labels = self.classifier.predict(self.testing_tweets)

        score = metrics.accuracy_score(self.testing_labels, predicted_labels)
        confusion_matrix = metrics.confusion_matrix(self.testing_labels, predicted_labels)
        f1_score = metrics.f1_score(self.testing_labels, predicted_labels, pos_label='4')
        recall = metrics.recall_score(self.testing_labels, predicted_labels, pos_label='4')
        precision = metrics.precision_score(self.testing_labels, predicted_labels, pos_label='4')

        return [score, confusion_matrix, f1_score, recall, precision]


    
