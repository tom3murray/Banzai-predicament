from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import re
import pandas as pd
from nltk.text import Text
from collections import Counter
from random import shuffle
from nltk.sentiment import SentimentIntensityAnalyzer
from pprint import pprint
import nltk
from nltk.corpus import state_union
import matplotlib.pyplot as plt
import os
from readability import Readability

# https://realpython.com/python-nltk-sentiment-analysis/#extracting-concordance-and-collocations

directory = r'/Users/wileysimonds/Desktop/School_Work/Current_Classes/PLAP_3270/Code/state_of_union'
stopwords = nltk.corpus.stopwords.words("english")

res = sorted(os.listdir(directory), key=lambda x: (len(x.split(' ')[0]), x.split(' ')[0]))
print(res)


optimizm_totals = []
len_sentences = []
average_positivitys = []
filenames = []
sentiment_data = pd.DataFrame()
for idx1, filename in enumerate(res[0:]):
    # print(idx1, filename)
    if filename.endswith(".txt"):
        filenames.append(filename)
        ########################################################################
        # Sentiment analysis of each sentence - graph this like a time plot - would be interesting to see if the total goes below zero and then comes up

        sia = SentimentIntensityAnalyzer()
        file = open("state_of_union/" + str(filename))
        file_content = file.read().replace("\n\n", " ").replace("\n", " ")
        file.close()
        x = []
        y = []
        total_tokenized_sentences = 0
        tokenized_sentences = sent_tokenize(file_content)
        for idx2, t in enumerate(tokenized_sentences):
            # print(t, "\n")
            polarity_score = sia.polarity_scores(t)
            # print(polarity_score, 'positivity ')
            total_tokenized_sentences += polarity_score['pos'] - \
                polarity_score['neg']  # += polarity_score['compound']
            x.append(idx2)
            y.append(total_tokenized_sentences)
        # print(total_tokenized_sentences, len(tokenized_sentences),
        #       total_tokenized_sentences/len(tokenized_sentences))

        num = filename.split(' ')[0]

        plt.plot(x, y, label=filename)

        optimizm_total = total_tokenized_sentences
        len_sentence = len(tokenized_sentences)
        average_positivity = total_tokenized_sentences / len_sentence
        optimizm_totals.append(optimizm_total)
        len_sentences.append(len_sentence)
        average_positivitys.append(average_positivity)

        # df_length = len(df)
        # sentiment_data.loc[len(sentiment_data)] = y
        # sentiment_data = sentiment_data.append({y}, ignore_index=True)
        # y_append = y.insert(0, 'hello there')
        sentiment_data = sentiment_data.append(
            pd.DataFrame([y]), ignore_index=True)
        ########################################################################
        ########################################################################
        # Remove stopwords and look at all of the most common words
        word_tokenized_file_content = nltk.word_tokenize(file_content.lower())
        most_common_non_stopwords = [w for w in word_tokenized_file_content if w not in stopwords]
        most_common_non_stopwords_punctuation = nltk.FreqDist(
            [w for w in most_common_non_stopwords if w.isalnum()])

        # print(most_common_non_stopwords_punctuation.most_common(10), 'most_common_non_stopwords')
        ########################################################################
        ########################################################################
        # Look at the most common nouns and proper nouns
        tags = nltk.pos_tag(word_tokenized_file_content)
        only_nn = [x for (x, y) in tags if y in ('NN', 'NNS', 'NNP', 'NNPS')]
        # print(only_nn, 'only_nn')

        freq = nltk.FreqDist(only_nn).most_common(10)
        # print(freq, 'freq')
        ########################################################################
        ########################################################################
        # BigramCollocationFinder, TrigramCollocationFinder, QuadgramCollocationFinder

        # words = [w.lower() for w in nltk.corpus.state_union.words() if w.isalpha()]
        #
        # word_tokenized_file_content_no_punctuation = nltk.FreqDist(
        #     [w for w in word_tokenized_file_content if w.isalnum()])

        bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(word_tokenized_file_content
                                                                             )
        trigram_finder = nltk.collocations.TrigramCollocationFinder.from_words(
            word_tokenized_file_content)
        quadgram_finder = nltk.collocations.QuadgramCollocationFinder.from_words(
            word_tokenized_file_content)

        # print(bigram_finder.ngram_fd.most_common(10))
        # print(trigram_finder.ngram_fd.most_common(10))
        # print(quadgram_finder.ngram_fd.most_common(10))
        ########################################################################
        ########################################################################
        # Readability
        r = Readability(file_content)
        fk = r.flesch_kincaid()
        print(fk.score, filename, fk.grade_level)
        ########################################################################

    else:
        pass


# Sentiment alanysis total of each
print(optimizm_totals, 'optimizm_totals')
print(len_sentences, 'len_sentences')
print(average_positivitys, 'average_positivitys')

# sentiment_data = sentiment_data.insert(loc=0, column='File Name', value=filenames)
# filenames
# sentiment_data = sentiment_data.reset_index()


print(filenames, len(filenames), 'filenames')
sentiment_data['Speech Name'] = filenames


cols = sentiment_data.columns.tolist()
cols = cols[-1:] + cols[:-1]
sentiment_data = sentiment_data[cols]

print(sentiment_data, 'sentiment_data')
sentiment_data.to_excel('./state_of_union_sentiment_data.xlsx', index=False)
plt.legend()
plt.show()


# Sentence length
# Are people consistently optimistic
# How does optimism change across a term
# What words are most correlated with optimism (God? War [negative correlation])
