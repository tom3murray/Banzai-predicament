import os
import pandas as pd
import spacy


os.chdir('../Banzai-predicament')

directory = '../Banzai-predicament/addresses/'

address_dataframe = pd.DataFrame(columns = ['address year', 'text'])

for filename in os.listdir(directory):
    h = os.path.join(directory, filename)
            
    if os.path.isfile(h):        

        with open(h, "r") as f:
            address_year = filename.split('.')[0]
            text = f.read()
            
            address_entry = [address_year, text]
            
            address_dataframe.loc[len(address_dataframe)] = address_entry

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 10000000
docs = list(nlp.pipe(address_dataframe['text'], disable = ['ner', 'parser']))

#%%
lemmas = []

for doc in docs:
    for lemma in doc:
        lemmas.append(lemma)
        
