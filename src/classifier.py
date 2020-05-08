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
    category_freq = Counter(category for category in labels)
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
    category_probability = category_probabilities.get(label, 1)
    # If the category hasn't been seen before the probability should be 0. So log(0) = -infinity
    # if category_probability == 0:
    #     return -math.inf
    if category_probability == 0:
        category_probability = 1

    score = np.log(category_probability)

    for word in document:
        word_prob = word_given_category_probabilities[label].get(word, 0)
        # If the word hasn't been seen before the probability should be 0. So log(0) = -infinity
        # if word_prob == 0:
        #      return -math.inf
        if word_prob == 0:
            word_prob = 1
        score += np.log(word_prob)
    return score


def classify_naive_bayes(document, word_given_category_probabilities, category_probabilities):
    class_and_score = dict()
    for category in category_probabilities.keys():
        score = score_doc_label(document, category, word_given_category_probabilities, category_probabilities)
        class_and_score.update({category: score})
    class_label = max(class_and_score, key=class_and_score.get)
    # Outputs the category and the score towards that category
    return class_label, class_and_score[class_label]


'''
    Task 3:
    Classifying all documents
    Give accuracy of the classifier
'''


def classify_documents(documents, word_given_category_probabilities, category_probabilities):
    doc_classification = []
    for document in documents:
        classification, class_and_score = classify_naive_bayes(document, word_given_category_probabilities,
                                                               category_probabilities)
        doc_classification.append(classification)
    return doc_classification


def accuracy(true_labels, guessed_labels):
    total_count = len(true_labels)
    missed = 0
    for true_label, guessed_label in zip(true_labels, guessed_labels):
        if true_label != guessed_label:
            missed += 1

    return (total_count - missed) / total_count


'''
    Task 4:
    Find the misclassified documents
    Comment why they were hard to classify
    
    The accuracy of this classifier is 0.6160302140159463.
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


def correct_classified_documents_with_label(true_labels, guessed_labels, documents):
    list_of_correct_classified_doc = []
    list_of_labels = []
    for true_label, guessed_label, document in zip(true_labels, guessed_labels, documents):
        if true_label == guessed_label:
            list_of_correct_classified_doc.append(document)
            list_of_labels.append(true_label)
    return list_of_correct_classified_doc, list_of_labels


def misclassified_documents_with_label(true_labels, guessed_labels, documents):
    list_of_misclassified_doc = []
    list_of_labels = []
    for true_label, guessed_label, document in zip(true_labels, guessed_labels, documents):
        if true_label != guessed_label:
            list_of_misclassified_doc.append(document)
            list_of_labels.append(guessed_label)
    return list_of_misclassified_doc, list_of_labels


'''
    Split the list into one for training and one for evaluation
'''

# all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')
#
# split_point = int(0.8 * len(all_docs))
# train_docs = all_docs[:split_point]
# train_labels = all_labels[:split_point]
# eval_docs = all_docs[split_point:]
# eval_labels = all_labels[split_point:]

train_docs, train_labels = read_documents('all_sentiment_shuffled.txt')
eval_docs, eval_labels = read_documents('Sample1 (remaster).txt')

word_cat_prob, cat_prob = train_naive_bayes(train_docs, train_labels)
guessed_doc_labels = classify_documents(eval_docs, word_cat_prob, cat_prob)

acc = accuracy(eval_labels, guessed_doc_labels)

list_of_incorrect_classification, list_of_incorrect_label = misclassified_documents_with_label(eval_labels,
                                                                                               guessed_doc_labels,
                                                                                               eval_docs)
list_of_correct_classification, list_of_correct_label = correct_classified_documents_with_label(eval_labels,
                                                                                                guessed_doc_labels,
                                                                                                eval_docs)

file_trained_set = open("data/listOfTrainedData.txt", "w")
file_trained_set.write("Format: " + str({"Probability": "Doc/Category"}) + "\n\n")
for cat in cat_prob:
    file_trained_set.write(str("=====NEW SECTION======") + "\n")
    file_trained_set.write(str({cat: cat_prob[cat]}) + "\n\n")
    for wrd in word_cat_prob[cat]:
        file_trained_set.write(str({wrd: word_cat_prob[cat][wrd]}) + "\n")
    file_trained_set.write("\n")
file_trained_set.close()

file_list_of_labels = open("data/listOfLabels.txt", "w")
file_list_of_labels.write("Format: " + str({"True": "Guessed"}) + "\n\n")
for true_lb, guessed_lb in zip(eval_labels, guessed_doc_labels):
    file_list_of_labels.write(str({true_lb: guessed_lb}) + "\n")
file_list_of_labels.write('The accuracy of the AI is: ' + str(acc))
file_list_of_labels.close()

file_list_of_doc_labels = open("data/listOfDocAndLabels.txt", "w")
file_list_of_doc_labels.write("Format: " + str({"True": "(Guessed, Score)"}) + " doc\n\n")
for true_lb, guessed_lb, doc in zip(eval_labels, guessed_doc_labels, eval_docs):
    topic_and_score = classify_naive_bayes(doc, word_cat_prob, cat_prob)
    file_list_of_doc_labels.write(str({true_lb: topic_and_score}) + str(doc) + "\n\n")
file_list_of_doc_labels.write('The accuracy of the AI is: ' + str(acc))
file_list_of_doc_labels.close()

file_list_of_correct_doc = open("data/listOfCorrectDoc.txt", "w")
file_list_of_correct_doc.write("Format: " + str({"Guessed": "Score"}) + " doc\n\n")
for doc, lbl in zip(list_of_correct_classification, list_of_correct_label):
    topic, score = classify_naive_bayes(doc, word_cat_prob, cat_prob)
    file_list_of_correct_doc.write(str({topic: score}) + str(doc) + "\n\n")
file_list_of_correct_doc.close()

file_list_of_incorrect_doc = open("data/listOfIncorrectDoc.txt", "w")
file_list_of_incorrect_doc.write("Format: " + str({"Guessed": "Score"}) + " doc\n\n")
for doc, lbl in zip(list_of_incorrect_classification, list_of_incorrect_label):
    topic, score = classify_naive_bayes(doc, word_cat_prob, cat_prob)
    file_list_of_incorrect_doc.write(str({topic: score}) + str(doc) + "\n\n")
file_list_of_incorrect_doc.close()
