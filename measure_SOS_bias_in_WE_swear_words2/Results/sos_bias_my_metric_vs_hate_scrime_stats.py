#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy import stats 
import pandas as pd


# In[3]:


min_df = pd.read_csv("./all_WE_minority_words_sim_to_profane_vector.csv")
maj_df = pd.read_csv("./all_WE_majority_words_sim_to_profane_vector.csv")


# In[4]:


min_df.columns


# In[5]:


maj_df.columns


# ## statisctical difference between the different groups: stigmatized, unstigmatized, and neutral words

# In[6]:


min_df.columns


# In[7]:


maj_df.columns


# ## lgtb

# In[8]:


lgtb_words = ["lesbian","gay","bisexual","transgender","tran","queer",
                "lgbt","lgbtq","homosexual","non-binary"]


# In[9]:


lgtbq_w2V_sim = min_df[min_df["ethnicity_word"].isin(lgtb_words)]["w2v_sim"]


# In[10]:


np.mean(lgtbq_w2V_sim)


# In[11]:


lgtbq_glove_wk_sim = min_df[min_df["ethnicity_word"].isin(lgtb_words)]["glove_wk_sim"]


# In[12]:


np.mean(lgtbq_glove_wk_sim)


# In[13]:


lgtbq_glove_twitter = min_df[min_df["ethnicity_word"].isin(lgtb_words)]["glove_twitter_sim"]


# In[14]:


lgtbq_ud = min_df[min_df["ethnicity_word"].isin(lgtb_words)]["ud_sim"]


# In[15]:


lgtbq_chan = min_df[min_df["ethnicity_word"].isin(lgtb_words)]["chan_sim"]


# In[16]:


np.mean(lgtbq_glove_twitter)


# In[17]:


np.mean(lgtbq_ud)


# In[18]:


np.mean(lgtbq_chan)


# In[19]:


stats.friedmanchisquare(lgtbq_w2V_sim, lgtbq_glove_wk_sim, lgtbq_glove_twitter, lgtbq_ud, lgtbq_chan)


# In[20]:


stats.ttest_ind(lgtbq_glove_wk_sim, lgtbq_w2V_sim)


# In[21]:


stats.ttest_ind(lgtbq_glove_wk_sim, lgtbq_glove_twitter)


# In[22]:


stats.ttest_ind(lgtbq_glove_wk_sim, lgtbq_ud)


# In[23]:


stats.ttest_ind(lgtbq_glove_wk_sim, lgtbq_chan)


# ## women

# In[24]:


women_words = ["woman", "female", "girl","wife","sister","daughter","mother"]


# In[25]:


women_w2v = min_df[min_df["ethnicity_word"].isin(women_words)]["w2v_sim"]
women_glove_wk = min_df[min_df["ethnicity_word"].isin(women_words)]["glove_wk_sim"]
women_glove_twitter = min_df[min_df["ethnicity_word"].isin(women_words)]["glove_twitter_sim"]
women_ud = min_df[min_df["ethnicity_word"].isin(women_words)]["ud_sim"]
women_chan = min_df[min_df["ethnicity_word"].isin(women_words)]["chan_sim"]


# In[26]:


np.mean(women_w2v)


# In[27]:


np.mean(women_glove_wk)


# In[28]:


np.mean(women_glove_twitter)


# In[29]:


np.mean(women_ud)


# In[30]:


np.mean(women_chan)


# In[31]:


stats.friedmanchisquare(women_w2v, women_glove_wk, women_glove_twitter, women_ud, women_chan)


# In[32]:


stats.ttest_ind(women_glove_twitter, women_w2v)


# In[33]:


stats.ttest_ind(women_glove_twitter, women_glove_wk)


# In[34]:


stats.ttest_ind(women_glove_twitter, women_ud)


# In[35]:


stats.ttest_ind(women_glove_twitter, women_chan)


# ## ethnicities

# In[36]:


eth_words = ["african", "african american", "asian", "black", "hispanic", "latin", "mexican", "indian", "middle eastern",
                "arab"]


# In[37]:


eth_w2v = min_df[min_df["ethnicity_word"].isin(eth_words)]["w2v_sim"]
eth_glove_wk = min_df[min_df["ethnicity_word"].isin(eth_words)]["glove_wk_sim"]
eth_glove_twitter = min_df[min_df["ethnicity_word"].isin(eth_words)]["glove_twitter_sim"]
eth_ud = min_df[min_df["ethnicity_word"].isin(eth_words)]["ud_sim"]
eth_chan = min_df[min_df["ethnicity_word"].isin(eth_words)]["chan_sim"]


# In[38]:


np.mean(eth_w2v)


# In[39]:


np.mean(eth_glove_wk)


# In[40]:


np.mean(eth_glove_twitter)


# In[41]:


np.mean(eth_ud)


# In[42]:


np.mean(eth_chan)


# In[43]:


stats.friedmanchisquare(eth_w2v, eth_glove_wk, eth_glove_twitter, eth_ud, eth_chan)


# In[44]:


stats.ttest_ind(eth_w2v, eth_glove_wk)


# In[45]:


stats.ttest_ind(eth_w2v, eth_glove_twitter)


# In[46]:


stats.ttest_ind(eth_w2v, eth_ud)


# In[47]:


stats.ttest_ind(eth_w2v, eth_chan)


# ## hate crime stats (department of justice US)

# In[96]:


crime_rate = [61.9, 15.4,0.7]


# In[97]:


stats.pearsonr([np.mean(eth_w2v), np.mean(lgtbq_w2V_sim), np.mean(women_w2v)], crime_rate)


# In[98]:


stats.pearsonr([np.mean(eth_glove_wk), np.mean(lgtbq_glove_wk_sim), np.mean(women_glove_wk)], crime_rate)


# In[99]:


stats.pearsonr([np.mean(eth_glove_twitter), np.mean(lgtbq_glove_twitter), np.mean(women_glove_twitter)], crime_rate)


# In[100]:


stats.pearsonr([np.mean(eth_ud), np.mean(lgtbq_ud), np.mean(women_ud)], crime_rate)


# In[101]:


stats.pearsonr([np.mean(eth_chan), np.mean(lgtbq_chan), np.mean(women_chan)], crime_rate )


# ## Hate crime England and wales

# In[54]:


crime_rate2 = [72, 8]


# In[55]:


stats.pearsonr([np.mean(eth_w2v), np.mean(lgtbq_w2V_sim)], crime_rate2)


# In[56]:


stats.pearsonr([np.mean(eth_glove_wk), np.mean(lgtbq_glove_wk_sim)], crime_rate2)


# In[57]:


stats.pearsonr([np.mean(eth_glove_twitter), np.mean(lgtbq_glove_twitter)], crime_rate2)


# In[58]:


stats.pearsonr([np.mean(eth_ud), np.mean(lgtbq_ud)], crime_rate2)


# In[59]:


stats.pearsonr([np.mean(eth_chan), np.mean(lgtbq_chan)], crime_rate2)


# ## Online Extremism and Online Hate

# In[90]:


hate_finland = [0.67, 0.63, 0.25]
hate_us = [0.60, 0.61, 0.44]
germany = [0.48, 0.50, 0.20]
uk = [0.57, 0.55, 0.44]


# In[62]:


np.mean(eth_w2v), np.mean(lgtbq_w2V_sim), np.mean(women_w2v)


# In[64]:


np.mean(eth_glove_wk), np.mean(lgtbq_glove_wk_sim), np.mean(women_glove_wk)


# In[91]:


print("w2v_finland",stats.pearsonr([np.mean(eth_w2v), np.mean(lgtbq_w2V_sim), np.mean(women_w2v)], hate_finland))
print("glove_wk_finland",stats.pearsonr([np.mean(eth_glove_wk), np.mean(lgtbq_glove_wk_sim), np.mean(women_glove_wk)], hate_finland))
print("glove_twitter",stats.pearsonr([np.mean(eth_glove_twitter), np.mean(lgtbq_glove_twitter), np.mean(women_glove_twitter)], hate_finland))
print("ud_finland",stats.pearsonr([np.mean(eth_ud), np.mean(lgtbq_ud), np.mean(women_ud)], hate_finland))
print("chan_finland",stats.pearsonr([np.mean(eth_chan), np.mean(lgtbq_chan), np.mean(women_chan)], hate_finland))


# In[92]:


print("w2v_us",stats.pearsonr([np.mean(eth_w2v), np.mean(lgtbq_w2V_sim), np.mean(women_w2v)], hate_us))
print("glove_wk_us",stats.pearsonr([np.mean(eth_glove_wk), np.mean(lgtbq_glove_wk_sim), np.mean(women_glove_wk)], hate_us))
print("glove_us",stats.pearsonr([np.mean(eth_glove_twitter), np.mean(lgtbq_glove_twitter), np.mean(women_glove_twitter)], hate_us))
print("ud_us",stats.pearsonr([np.mean(eth_ud), np.mean(lgtbq_ud), np.mean(women_ud)], hate_us))
print("chan_us",stats.pearsonr([np.mean(eth_chan), np.mean(lgtbq_chan), np.mean(women_chan)], hate_us))


# In[93]:


print("w2v_germany",stats.pearsonr([np.mean(eth_w2v), np.mean(lgtbq_w2V_sim), np.mean(women_w2v)], germany))
print("glove_wk_germany",stats.pearsonr([np.mean(eth_glove_wk), np.mean(lgtbq_glove_wk_sim), np.mean(women_glove_wk)], germany))
print("glove_germany",stats.pearsonr([np.mean(eth_glove_twitter), np.mean(lgtbq_glove_twitter), np.mean(women_glove_twitter)], germany))
print("ud_germany",stats.pearsonr([np.mean(eth_ud), np.mean(lgtbq_ud), np.mean(women_ud)], germany))
print("chan_germany",stats.pearsonr([np.mean(eth_chan), np.mean(lgtbq_chan), np.mean(women_chan)], germany))


# In[94]:


print("w2v_uk",stats.pearsonr([np.mean(eth_w2v), np.mean(lgtbq_w2V_sim), np.mean(women_w2v)], uk))
print("glove_wk_uk",stats.pearsonr([np.mean(eth_glove_wk), np.mean(lgtbq_glove_wk_sim), np.mean(women_glove_wk)], uk))
print("glove_uk",stats.pearsonr([np.mean(eth_glove_twitter), np.mean(lgtbq_glove_twitter), np.mean(women_glove_twitter)], uk))
print("ud_uk",stats.pearsonr([np.mean(eth_ud), np.mean(lgtbq_ud), np.mean(women_ud)], uk))
print("chan_uk",stats.pearsonr([np.mean(eth_chan), np.mean(lgtbq_chan), np.mean(women_chan)], uk))


# In[71]:


## difference between majoirty and minority added (online harassment) (https://www.womensmediacenter.com/speech-project/research-statistics)


# In[72]:


bullying_diff = [0.23,0.25,0.31]


# In[73]:


print("w2v_uk",stats.pearsonr([np.mean(eth_w2v), np.mean(lgtbq_w2V_sim), np.mean(women_w2v)], bullying_diff))
print("glove_wk_uk",stats.pearsonr([np.mean(eth_glove_wk), np.mean(lgtbq_glove_wk_sim), np.mean(women_glove_wk)], bullying_diff))
print("glove_uk",stats.pearsonr([np.mean(eth_glove_twitter), np.mean(lgtbq_glove_twitter), np.mean(women_glove_twitter)], bullying_diff))
print("ud_uk",stats.pearsonr([np.mean(eth_ud), np.mean(lgtbq_ud), np.mean(women_ud)], bullying_diff))
print("chan_uk",stats.pearsonr([np.mean(eth_chan), np.mean(lgtbq_chan), np.mean(women_chan)], bullying_diff))


# In[93]:


## https://enough.org/stats_cyberbullying


# In[94]:


bullying = [0.17,0.15,0.2]


# In[95]:


print("w2v_uk",stats.pearsonr([np.mean(eth_w2v), np.mean(lgtbq_w2V_sim), np.mean(women_w2v)], bullying))
print("glove_wk_uk",stats.pearsonr([np.mean(eth_glove_wk), np.mean(lgtbq_glove_wk_sim), np.mean(women_glove_wk)], bullying))
print("glove_uk",stats.pearsonr([np.mean(eth_glove_twitter), np.mean(lgtbq_glove_twitter), np.mean(women_glove_twitter)], bullying))
print("ud_uk",stats.pearsonr([np.mean(eth_ud), np.mean(lgtbq_ud), np.mean(women_ud)], bullying))
print("chan_uk",stats.pearsonr([np.mean(eth_chan), np.mean(lgtbq_chan), np.mean(women_chan)], bullying))


# In[ ]:




