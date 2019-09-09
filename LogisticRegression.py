# Henry Dinh
# CS 6375.001
# Assignment 2 - Logistic Regression algorithm
# To test the program, read the README file for instructions

import os
import sys
import collections
import re
import math
import copy

# Stores emails as dictionaries. email_file_name : Document (class defined below)
training_set = dict()
test_set = dict()

# Filtered sets without stop words
filtered_training_set = dict()
filtered_test_set = dict()

# list of Stop words
stop_words = []

# Vocabulary/tokens in the training set
training_set_vocab = []
filtered_training_set_vocab = []

# store weights as dictionary. w0 initiall 0.0, others initially 0.0. token : weight value
weights = {'weight_zero': 0.0}
filtered_weights = {'weight_zero': 0.0}

# ham = 0 for not spam, spam = 1 for is spam
classes = ["ham", "spam"]

# Natural learning rate constant, number of iterations for learning weights, and penalty (lambda) constant
learning_constant = .001
num_iterations = 100
penalty = 0.0

# Read all text files in the given directory and construct the data set, D
# the directory path should just be like "train/ham" for example
# storage is the dictionary to store the email in
# True class is the true classification of the email (spam or ham)
def makeDataSet(storage_dict, directory, true_class):
    for dir_entry in os.listdir(directory):
        dir_entry_path = os.path.join(directory, dir_entry)
        if os.path.isfile(dir_entry_path):
            with open(dir_entry_path, 'r') as text_file:
                # stores dictionary of dictionary of dictionary as explained above in the initialization
                text = text_file.read()
                storage_dict.update({dir_entry_path: Document(text, bagOfWords(text), true_class)})


# Extracts the vocabulary of all the text in a data set
def extractVocab(data_set):
    v = []
    for i in data_set:
        for j in data_set[i].getWordFreqs():
            if j not in v:
                v.append(j)
    return v


# Set the stop words
def setStopWords():
    stops = []
    with open('stop_words.txt', 'r') as txt:
        stops = (txt.read().splitlines())
    return stops


# Remove stop words from data set and store in dictionary
def removeStopWords(stops, data_set):
    filtered_data_set = copy.deepcopy(data_set)
    for i in stops:
        for j in filtered_data_set:
            if i in filtered_data_set[j].getWordFreqs():
                del filtered_data_set[j].getWordFreqs()[i]
    return filtered_data_set


# counts frequency of each word in the text files and order of sequence doesn't matter
def bagOfWords(text):
    bagsofwords = collections.Counter(re.findall(r'\w+', text))
    return dict(bagsofwords)


# Learn weights by using gradient ascent
def learnWeights(training, weights_param, iterations, lam):
    # Adjust weights num_iterations times
    for x in range(0, iterations):
        print x
        # Adjust each weight...
        counter = 1
        for w in weights_param:
            sum = 0.0
            # ...using all training instances
            for i in training:
                # y_sample is true y value (classification) of the doc
                y_sample = 0.0
                if training[i].getTrueClass() == classes[1]:
                    y_sample = 1.0
                # Only add to the sum if the doc contains the token (the count of it would be 0 anyways)
                if w in training[i].getWordFreqs():
                    sum += float(training[i].getWordFreqs()[w]) * (y_sample - calculateCondProb(classes[1], weights_param, training[i]))
            weights_param[w] += ((learning_constant * sum) - (learning_constant * float(lam) * weights_param[w]))


# Calculate conditional probability for the specified doc. Where class_prob is 1|X or 0|X
# 1 is spam and 0 is ham
def calculateCondProb(class_prob, weights_param, doc):
    # Total tokens in doc. Used to normalize word counts to stay within 0 and 1 for avoiding overflow
    # total_tokens = 0.0
    # for i in doc.getWordFreqs():
    #     total_tokens += doc.getWordFreqs()[i]

    # Handle 0
    if class_prob == classes[0]:
        sum_wx_0 = weights_param['weight_zero']
        for i in doc.getWordFreqs():
            if i not in weights_param:
                weights_param[i] = 0.0
            # sum of weights * token count for each token in each document
            sum_wx_0 += weights_param[i] * float(doc.getWordFreqs()[i])
        return 1.0 / (1.0 + math.exp(float(sum_wx_0)))

    # Handle 1
    elif class_prob == classes[1]:
        sum_wx_1 = weights_param['weight_zero']
        for i in doc.getWordFreqs():
            if i not in weights_param:
                weights_param[i] = 0.0
            # sum of weights * token count for each token in each document
            sum_wx_1 += weights_param[i] * float(doc.getWordFreqs()[i])
        return math.exp(float(sum_wx_1)) / (1.0 + math.exp(float(sum_wx_1)))


# Apply algorithm to guess class for specific instance of test set
def applyLogisticRegression(data_instance, weights_param):
    score = {}
    score[0] = calculateCondProb(classes[0], weights_param, data_instance)
    score[1] = calculateCondProb(classes[1], weights_param, data_instance)
    if score[1] > score[0]:
        return classes[1]
    else:
        return classes[0]



# Document class to store email instances easier
class Document:
    text = ""
    # x0 assumed 1 for all documents (training examples)
    word_freqs = {'weight_zero': 1.0}

    # spam or ham
    true_class = ""
    learned_class = ""

    # Constructor
    def __init__(self, text, counter, true_class):
        self.text = text
        self.word_freqs = counter
        self.true_class = true_class

    def getText(self):
        return self.text

    def getWordFreqs(self):
        return self.word_freqs

    def getTrueClass(self):
        return self.true_class

    def getLearnedClass(self):
        return self.learned_class

    def setLearnedClass(self, guess):
        self.learned_class = guess


# takes directories holding the data text files as paramters. "train/ham" for example
def main(training_spam_dir, training_ham_dir, test_spam_dir, test_ham_dir, lambda_constant):
    # Set up data sets. Dictionaries containing the text, word frequencies, and true/learned classifications
    makeDataSet(training_set, training_spam_dir, classes[1])
    makeDataSet(training_set, training_ham_dir, classes[0])
    makeDataSet(test_set, test_spam_dir, classes[1])
    makeDataSet(test_set, test_ham_dir, classes[0])
    penalty = lambda_constant

    # Set the stop words list
    stop_words = setStopWords()

    # Set up data sets without stop words
    filtered_training_set = removeStopWords(stop_words, training_set)
    filtered_test_set = removeStopWords(stop_words, test_set)

    # Extract training set vocabulary
    training_set_vocab = extractVocab(training_set)
    filtered_training_set_vocab = extractVocab(filtered_training_set)

    # Set all weights in training set vocabulary to be initially 0.0. w0 ('weight_zero') is initially 0.0
    for i in training_set_vocab:
        weights[i] = 0.0
    for i in filtered_training_set_vocab:
        filtered_weights[i] = 0.0

    # Learn weights
    learnWeights(training_set, weights, num_iterations, penalty)
    learnWeights(filtered_training_set, filtered_weights, num_iterations, penalty)


    # Apply algorithm on test set
    correct_guesses = 0.0
    for i in test_set:
        test_set[i].setLearnedClass(applyLogisticRegression(test_set[i], weights))
        if test_set[i].getLearnedClass() == test_set[i].getTrueClass():
            correct_guesses += 1.0

    # Apply algorithm on filtered test set
    correct_guesses_filtered = 0.0
    for i in filtered_test_set:
        filtered_test_set[i].setLearnedClass(applyLogisticRegression(filtered_test_set[i], filtered_weights))
        if filtered_test_set[i].getLearnedClass() == filtered_test_set[i].getTrueClass():
            correct_guesses_filtered += 1.0

    print "Correct guesses before filtering stop words:\t%d/%s" % (correct_guesses, len(test_set))
    print "Accuracy before filtering stop words:\t\t\t%.4f%%" % (100.0 * float(correct_guesses) / float(len(test_set)))
    print
    print "Correct guesses after filtering stop words:\t\t%d/%s" % (correct_guesses_filtered, len(filtered_test_set))
    print "Accuracy after filtering stop words:\t\t\t%.4f%%" % (100.0 * float(correct_guesses_filtered) / float(len(filtered_test_set)))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])