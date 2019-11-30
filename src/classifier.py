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
    category_probability = category_probabilities.get(label, 0)
    if category_probability == 0:
        return -math.inf
    score = np.log(category_probability)

    for word in document:
        word_prob = word_given_category_probabilities[label].get(word, 0)
        if word_prob == 0:
            return -math.inf
        else:
            score += np.log(word_prob)
    return score


def classify_naive_bayes(document, word_given_category_probabilities, category_probabilities):
    class_and_score = dict()
    for category in category_probabilities.keys():
        score = score_doc_label(document, category, word_given_category_probabilities, category_probabilities)
        class_and_score.update({category: score})
    class_label = max(class_and_score, key=class_and_score.get)
    return class_label, class_and_score[class_label]


'''
    Task 3:
    Classifying all documents
    Give accuracy of the classifier
'''


def classify_documents(documents, word_given_category_probabilities, category_probabilities):
    class_and_doc = dict()
    for document in documents:
        classification, class_and_score = classify_naive_bayes(document, word_given_category_probabilities,
                                                               category_probabilities)
        class_and_doc.setdefault(classification, []).append(document)
    return class_and_doc


def accuracy(true_labels, guessed_labels):
    total_count = 0
    missed = 0
    for label in true_labels.keys():
        total_count += len(true_labels[label])
        true_label_docs = true_labels[label]
        for guessed_label_doc in guessed_labels[label]:
            if guessed_label_doc not in true_label_docs:
                missed += 1
    return (total_count-missed) / total_count


'''
    Task 4:
    Find the misclassified documents
    Comment why they were hard to classify
    
    The accuracy of this classifier is 0.6168694922366764.
    The main reason the classifier had some difficulties to classify some documents is simply because
        the machine learning algorithm (Naive Bayes Classifier) did not recognize some words that were
        in the documents it had to evaluate. In other words, the AI will not classify properly if it
        trained with some list of words, but then evaluates some document with words it did not trained on.
        That's because the AI didn't calculate any probability for words given a class it did not see before.
        Because of that, the probability that of an unknown word in its vocabulary will always result in 0, or
        log(0) = -infinity.
        And, since Naive Bayes assumes that all words are independent, the product of all words in the document that
        contains an unknown word in its vocabulary will result in 0. Hence, the document will be 
        classified "arbitrarily" depending on implementation. In my case, the default category will always be the 
        "first" category which is 'neg'. 
        Since the score of 0 or log(0) will be assigned to the document for all categories, the program will simply 
        pick the "first" category it encountered (i.e. "class_label = max(class_and_score, key=class_and_score.get)").
        
    The other reason the classifier has come difficulties is because, while the review might be negative, there were
        some positive words in the document. For example, "This is a good charger-quick and portable. But the battery
        life of the battery sucks!" is a negative review, but was classified as positive. There are both "pos" and "neg"
        words, so the classifier will obviously have some trouble classifying this document correctly.  
'''


def misclassified_documents(true_labels, guessed_labels):
    list_of_misclassified_doc = dict()
    for label in true_labels.keys():
        true_label_docs = true_labels[label]
        for guessed_label_doc in guessed_labels[label]:
            if guessed_label_doc not in true_label_docs:
                list_of_misclassified_doc.setdefault(label, []).append(guessed_label_doc)
    return list_of_misclassified_doc


'''
    Split the list into one for training and one for evaluation
'''

all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')

split_point = int(0.8 * len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

word_cat_prob, cat_prob = train_naive_bayes(train_docs, train_labels)
guessed__doc_labels = classify_documents(eval_docs, word_cat_prob, cat_prob)

true_doc_labels = dict()
for cat, doc in zip(eval_labels, eval_docs):
    true_doc_labels.setdefault(cat, []).append(doc)

accuracy = accuracy(true_doc_labels, guessed__doc_labels)
list_of_incorrect_class = misclassified_documents(true_doc_labels, guessed__doc_labels)

print('===============================LIST OF MISCLASSIFIED DOCUMENTS===============================')
for lbl in list_of_incorrect_class.keys():
    for doc in list_of_incorrect_class[lbl]:
        topic_label, score_label = classify_naive_bayes(doc, word_cat_prob, cat_prob)
        print({lbl: doc})
        print({topic_label: score_label})
        print()

print('The accuracy of the AI is: ' + str(accuracy))
