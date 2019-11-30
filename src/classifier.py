import math
from collections import Counter
import numpy as np

'''
    Task 0: 
    Read the file (test subject)
    Remove document identifier and topic labels.
    Prepare a list of all words docs
    Prepare a list of sentiment label
'''


def read_documents(doc_file):
    docs = []
    labels = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
    return docs, labels


'''
    Task 1:
    Labels are categories
    Documents are words
    Getting the parameters for naive bayes
'''


def train_naive_bayes(documents, labels):
    # Smoothing value
    smoothing = 0.5

    # Get the total list of words in context
    total_words = set(word for document in documents for word in document)

    # Organizes documents given the category
    docs_given_category = dict()
    for category, document in zip(labels, documents):
        docs_given_category.setdefault(category, []).append(document)

    # Organizes occurrences of a word given the category
    counter_of_words_given_category = dict()
    for category in docs_given_category:
        for document in docs_given_category[category]:
            word_counter = Counter(word for word in document)
            counter_of_words_given_category.setdefault(category, Counter()).update(word_counter)

    # Probability of every word given the category with smoothing
    word_given_category_probabilities = dict()
    for category in counter_of_words_given_category:
        sum_of_all_words_in_category = sum(counter_of_words_given_category[category].values())
        occurrence_of_words_in_category = dict(counter_of_words_given_category[category])
        for word in total_words:
            word_probability = ((occurrence_of_words_in_category.get(word, 0) + smoothing)
                                / (sum_of_all_words_in_category + (smoothing * len(total_words))))
            word_given_category_probabilities.setdefault(category, dict()).update({word: word_probability})

    # Computes the probability of the categories P(c_i)
    category_probabilities = dict()
    category_freq = Counter(category for category in all_labels)
    for category in category_freq:
        category_probabilities.update({category: (category_freq[category] / sum(category_freq.values()))})

    return word_given_category_probabilities, category_probabilities


'''
    Task 2:
    Getting score of the document given the label
    If the word or label were not trained, then return -infinity because limit(log(0)) is -infinity
    Define a classifier that would classify a document based on its words and label
'''


def score_doc_label(document, label, word_given_category_probabilities, category_probabilities):
    cat_prob = category_probabilities.get(label, 0)
    if cat_prob == 0:
        return -math.inf

    score = np.log(cat_prob)
    for word in document:
        word_prob = word_given_category_probabilities[label].get(word, 0)
        if word_prob == 0:
            return -math.inf
        else:
            score += np.log(word_prob)
    return score


def classify_naive_bayes(document, word_given_category_probabilities, category_probabilities):
    class_and_prob = dict()
    for category in category_probabilities.keys():
        score = score_doc_label(document, category, word_given_category_probabilities, category_probabilities)
        class_and_prob.update({category: score})
    return max(class_and_prob, key=class_and_prob.get)


'''
    Task 3:
    Classifying all documents
    Give accuracy of the classifier
'''


def classify_documents(docs, word_given_category_probabilities, category_probabilities):
    class_and_doc = dict()
    for doc in docs:
        classification = classify_naive_bayes(doc, word_given_category_probabilities, category_probabilities)
        class_and_doc.update({classification: doc})
    return class_and_doc


def accuracy(true_labels, guessed_labels):
    return None


'''
    Testing the Counter method
'''

example_documents = ['the first document'.split(), 'the second document'.split()]
freq1 = Counter(w for doc in example_documents for w in doc)
print(freq1)

'''
    Split the list into one for training and one for evaluation
'''

all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')

split_point = int(0.8 * len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

count = Counter({'red': 1, 'blue': 2})
count.update(Counter({'red': 1}))
c = dict(count)
print(count)
