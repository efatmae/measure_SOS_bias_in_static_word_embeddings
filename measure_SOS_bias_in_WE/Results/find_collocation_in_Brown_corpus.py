#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.collocations import *


# In[2]:


brown_corpus = nltk.corpus.brown.words(categories=['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies',
'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance',
'science_fiction'])


# In[7]:


corpus_name= "brown_corpus"


# In[ ]:


reddit_file = open("../../../Data/Reddit/reddit_body.txt", 'r').read()
reddit_list = reddit_file.split()
reddit_corpus = nltk.Text(reddit_list)


# In[3]:


wanted_word1 = 'girl'
wanted_word2 = ''


# In[4]:


bigram_measures = nltk.collocations.BigramAssocMeasures()
word1 = []
word2 = []
pmi = []

# Ngrams with 'creature' as a member
creature_filter1 = lambda *w: wanted_word not in w
creature_filter2 = lambda *w: wanted_word2 not in w
ignored_words = nltk.corpus.stopwords.words('english')


## Bigrams
finder = BigramCollocationFinder.from_words(brown_corpus)
# only bigrams that appear 3+ times
finder.apply_freq_filter(0)
# only bigrams that contain 'creature'
finder.apply_ngram_filter(creature_filter1)
#finder.apply_ngram_filter(creature_filter2)
finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
# return the 10 n-grams with the highest PMI
for i in finder.score_ngrams(bigram_measures.pmi):
    word1.append(i[0][0])
    word2.append(i[0][1])
    pmi.append(i[1])


# In[9]:


import pandas as pd
pd.DataFrame({"word1":word1, "word2":word2, "PMI":pmi}).to_csv("./"+corpus_name+"_"+wanted_word+"_collocation.csv")


# In[ ]:




