import nltk
from nltk import NaiveBayesClassifier
import random

# Import the data from preprocess.py file
from preprocess import ham_list, spam_list

# Randomize the data
combined_list = ham_list + spam_list
random.shuffle(combined_list)

# Split the train and test data
training_part = int(len(combined_list) * .7)

train_set = combined_list[:training_part]
test_set =  combined_list[training_part:]
 
print (len(train_set))
print (len(test_set))

# Create the Naive Bayes filter
classifier = NaiveBayesClassifier.train(train_set)
 
# Find the accuracy, using the test data
accuracy = nltk.classify.util.accuracy(classifier, test_set)
print("Accuracy is: ", accuracy * 100)

# See the most informative features
print(classifier.show_most_informative_features(20))
