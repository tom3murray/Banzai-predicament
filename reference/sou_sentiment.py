import re
import pandas as pd
from nltk.text import Text
from collections import Counter
from random import shuffle
from nltk.sentiment import SentimentIntensityAnalyzer
from pprint import pprint
import nltk
from nltk.corpus import state_union


# https://realpython.com/python-nltk-sentiment-analysis/#extracting-concordance-and-collocations


# nltk.download(["state_union", "stopwords"])
# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.corpus.state_union

# words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]
# print(words, 'words')
# stopwords = nltk.corpus.stopwords.words("english")
# print(stopwords, 'stopwords')
# words = [w for w in words if w.lower() not in stopwords]
# print(words)


# Look at the positivity of all the words
# Remove stopwords and look at all of the most common words
# Look at the most common nouns and proper nouns
# BigramCollocationFinder, TrigramCollocationFinder, QuadgramCollocationFinder
# Sentiment analysis of each sentence accross the speech graphed
# Sentiment alanysis of each speech in total
# Parts of speech
# Most commonly used words
# Reading level

################################################################################
################################################################################
################################################################################
# text = nltk.Text(nltk.corpus.state_union.words())
# concordance_list = text.concordance_list("america", lines=2)
# for entry in concordance_list:
#     print(entry.line)
################################################################################
################################################################################
################################################################################
# words = [w.lower() for w in nltk.corpus.state_union.words() if w.isalpha()]
#
# bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(words)
# trigram_finder = nltk.collocations.TrigramCollocationFinder.from_words(words)
# quadgram_finder = nltk.collocations.QuadgramCollocationFinder.from_words(words)
#
# print(bigram_finder.ngram_fd.most_common(16))
# print(trigram_finder.ngram_fd.most_common(16))
# print(quadgram_finder.ngram_fd.most_common(16))
# finder.ngram_fd.tabulate(10)
################################################################################
################################################################################
################################################################################
#
# def is_positive(tweet: str) -> bool:
#     """True if tweet has positive compound sentiment, False otherwise."""
#     return sia.polarity_scores(tweet)["compound"] > 0
#
#
# sia = SentimentIntensityAnalyzer()
# print(sia.polarity_scores("Wow, NLTK is really powerful!"))
#
# shuffle(tweets)
# for tweet in tweets[:10]:
#     print(">", is_positive(tweet), tweet)


################################################################################
################################################################################
################################################################################
# print(state_union.fileids())
# print(state_union.raw('2006-GWBush.txt'))


def analyzeTextForPOS(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text.lower())
    text = nltk.Text(tokens)
    # get parts of speech for each token
    tags = nltk.pos_tag(text)
    # count how many times each pos is used
    counts = Counter(tag for word, tag in tags)
    # note that the POS abbreviations can be understood here:
    # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    # return the counts as a dictionary
    return(dict(counts))


analyzeTextForPOS(state_union.raw('2006-GWBush.txt'))


def analyzeTexts():

    # Create a pattern for the regular expression (regex)
    # we're going to do further on.

    pattern = "(\d+)\-([A-Za-z]+)\-?(\d?)\.txt"
    # This pattern captures the date:     (\d+)
    # Then looks for a dash:              \-
    # Then captures a name:               ([A-Za-z]+)
    # Then looks for an optional dash:    \-?
    # Then captures an optional number:   (\d?)
    # Then looks for the .txt ending:     \.txt

    all_basic_info = []  # list of basic info dicts
    all_pos_info = []    # list of part of speech dicts
    all_totals = []      # list of total POS counted dicts

    # Note: we create three lists so that when we assemble
    # a data frame, these three types of information can be
    # placed in order, with basic info followed by pos counts
    # followed by total.  Since dicts are inherently unordered,
    # if we did a single list, we'd end up with columns in
    # alphabetical order, which is not what we want.

    # iterate over files and extract and organize data

    for filename in state_union.fileids():
        address_details = {}    # empty dict
        pos = {}                # empty dict
        totals = {}             # empty dict

        # Below, we're going to use  regular expressions to
        # extract meaningful info from the file name.

        m = re.match(pattern, filename)
        if m:
            address_details['address_date'] = m.group(1)
            address_details['president_name'] = m.group(2)
            address_details['optional_sequence'] = m.group(3)
            all_basic_info.append(address_details)  # add details of this file

            pos = analyzeTextForPOS(state_union.raw(filename))
            all_pos_info.append(pos)                # add part of speech for this file

            totals['Total'] = sum(pos.values())
            all_totals.append(totals)               # add total POS for this file

        else:
            print("Wonky problem with filename " + filename)

    return(pd.DataFrame(all_basic_info).join(pd.DataFrame(all_pos_info).join(pd.DataFrame(all_totals))))
    # return the results in a data frame -- leftmost columns are basic info,
    # then the part of speech info, then the totals in the rightmost column.


sotu_data = analyzeTexts()

print(sotu_data, 'sotu_data')

sotu_by_president = sotu_data.loc[:, 'president_name':]
sotu_by_president = sotu_by_president.groupby('president_name').sum()
# cols_i_want = ['JJR', 'JJS', 'RBR', 'RBS', 'Total']
# sotu_by_president = sotu_by_president.loc[:, cols_i_want]
republicans = ['Bush', 'Eisenhower', 'Ford', 'Nixon', 'Reagan', 'GWBush']
democrats = ['Carter', 'Clinton', 'Johnson', 'Kennedy', 'Truman']

republican_speech = sotu_by_president.loc[sotu_by_president.index.isin(republicans)]
democrat_speech = sotu_by_president.loc[sotu_by_president.index.isin(democrats)]
democrat_speech = democrat_speech.div(democrat_speech['Total'], axis=0)
republican_speech = republican_speech.div(republican_speech['Total'], axis=0)

print(republican_speech, 'republican_speech')
print(democrat_speech, 'democrat_speech')
