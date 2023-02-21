
import os
import pandas as pd
import matplotlib.pyplot as plt
import shifterator as sh
from collections import defaultdict
import string
import nltk
from nltk.corpus import stopwords


#%%
# CODE FOR COMPARING SENTIMENT AVG SCORES OF 2 SPEECHES + WORD SHIFT GRAPH 


os.chdir('C:/Users/Lyle_/Desktop/4 Yr Sem 2/DS')
os.getcwd()
df = pd.read_excel('all_name_data.xlsx')



# Set english stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))



# Read in labMT dictionary
# Source: https://github.com/ryanjgallagher/shifterator/blob/master/shifterator/lexicons/labMT/labMT_English.tsv
labmt = pd.read_excel('labMT_English.xlsx')
type2score_1 = pd.Series(labmt.score.values,index=labmt.word).to_dict()



text = df.iloc[0,2]
# Source: https://www.geeksforgeeks.org/python-remove-punctuation-from-string/
text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
text = text.lower()
words = text.split()
# Remove stop words / remove 'neutral' words
# Source: https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
clean_words = [word for word in words if not word in stop_words]
clean_words = [word for word in clean_words if not ((word in type2score_1) and (4 < type2score_1[word] < 6))]
# Source: https://www.geeksforgeeks.org/defaultdict-in-python/
type2freq_1 = defaultdict(int)
for word in clean_words:
    type2freq_1[word] += 1





text2 = df.iloc[233,2]
# Source: https://www.geeksforgeeks.org/python-remove-punctuation-from-string/
text2 = text2.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
text2 = text2.lower()
words2 = text2.split()
# Remove stop words
# Source: https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
clean_words2 = [word for word in words2 if not word in stop_words]
clean_words2 = [word for word in clean_words2 if not ((word in type2score_1) and (4 < type2score_1[word] < 6))]
# Source: https://www.geeksforgeeks.org/defaultdict-in-python/
type2freq_2 = defaultdict(int)
for word in clean_words2:
    type2freq_2[word] += 1
    



# Calculate weighted average word shift
# Source: https://shifterator.readthedocs.io/en/latest/cookbook/weighted_avg_shifts.html
sentiment_shift = sh.WeightedAvgShift(type2freq_1=type2freq_1,
                                      type2freq_2=type2freq_2,
                                      type2score_1=type2score_1,
                                      reference_value=5,
                                      stop_lens=[(4,6)])

# Graph differences
sentiment_shift.get_shift_graph(detailed=True, 
                                system_names=['Washington', 'Trump'])




# Get raw average sentiment scores
avg1 = sentiment_shift.get_weighted_score(type2freq_1, type2score_1)
print(avg1)
avg2 = sentiment_shift.get_weighted_score(type2freq_2, type2score_1)
print(avg2)


#%%
# CODE FOR GETTING ALL SENTIMENT AVG SCORES

df2 = []
for i in range(len(df.index)):
    
    # Clean and create dictionary for i speech
    text = df.iloc[i,2]
    # Source: https://www.geeksforgeeks.org/python-remove-punctuation-from-string/
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = text.lower()
    words = text.split()
    # Remove stop words / remove 'neutral' words
    # Source: https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
    clean_words = [word for word in words if not word in stop_words]
    clean_words = [word for word in clean_words if not ((word in type2score_1) and (4 < type2score_1[word] < 6))]
    # Source: https://www.geeksforgeeks.org/defaultdict-in-python/
    type2freq_1 = defaultdict(int)
    for word in clean_words:
        type2freq_1[word] += 1
    
    
    
    
    # Clean and create dictionary for set, base speech for comparison
    text2 = df.iloc[0,2]
    # Source: https://www.geeksforgeeks.org/python-remove-punctuation-from-string/
    text2 = text2.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text2 = text2.lower()
    words2 = text2.split()
    # Remove stop words
    # Source: https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
    clean_words2 = [word for word in words2 if not word in stop_words]
    clean_words2 = [word for word in clean_words2 if not ((word in type2score_1) and (4 < type2score_1[word] < 6))]
    # Source: https://www.geeksforgeeks.org/defaultdict-in-python/
    type2freq_2 = defaultdict(int)
    for word in clean_words2:
        type2freq_2[word] += 1
    
    
    
    # Sentiment comparison between the i and base speech
    sentiment_shift = sh.WeightedAvgShift(type2freq_1=type2freq_1,
                                          type2freq_2=type2freq_2,
                                          type2score_1=type2score_1,
                                          reference_value=5,
                                          stop_lens=[(4,6)])
    
    # Calculate average sentiment
    avg1 = sentiment_shift.get_weighted_score(type2freq_1, type2score_1)
    
    # Add score into df
    df2.append(
        {
            "President": df.iloc[i,0],
            "Date": df.iloc[i,1],
            "Sentiment": avg1
            }
        )
    
df2 = pd.DataFrame(df2)
# df2.to_excel(r'C:/Users/Lyle_/Desktop/4 Yr Sem 2/DS/basic_sentiment_new.xlsx', index=False)



# Graph all sentiment scores
df2.reset_index(inplace=True)


xaxis = df2["index"] +1790
yaxis = df2["Sentiment"]
plt.plot(xaxis, yaxis)
plt.title('Sentiment')
