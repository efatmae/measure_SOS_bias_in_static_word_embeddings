#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import stats 
import pandas as pd


# ## WEAT_SOS_bias_scores

# In[2]:


weat_sos_women_w2v = 0.046077
weat_sos_eth_w2v = 0.636393
weat_sos_lgtbq_w2v = 0.824543


# In[3]:


weat_sos_women_glove_wk = 0.051040
weat_sos_eth_glove_wk = 0.302182
weat_sos_lgtbq_glove_wk = 0.001300


# In[4]:


weat_sos_women_glove_twitter = 0.054989
weat_sos_eth_glove_twitter = 0.189505
weat_sos_lgtbq_glove_twitter = 0


# In[5]:


weat_sos_women_ud = 0.032825
weat_sos_eth_ud = 0.001476
weat_sos_lgtbq_ud = 0.412504


# In[6]:


weat_sos_women_chan = 0.114028
weat_sos_eth_chan = 0.441893
weat_sos_lgtbq_chan = 0.593770


# ## hate crime stats (department of justice US)

# In[7]:


crime_rate = [61.9, 15.4,0.7]


# In[8]:


stats.pearsonr([weat_sos_eth_w2v, weat_sos_lgtbq_w2v, weat_sos_women_w2v], crime_rate)


# In[9]:


stats.pearsonr([weat_sos_eth_glove_wk, weat_sos_lgtbq_glove_wk, weat_sos_women_glove_wk], crime_rate)


# In[10]:


#stats.pearsonr([weat_sos_eth_glove_twitter, weat_sos_lgtbq_glove_twitter, weat_sos_women_glove_twitter], crime_rate)


# In[11]:


stats.pearsonr([weat_sos_eth_ud, weat_sos_lgtbq_ud, weat_sos_women_ud], crime_rate)


# In[12]:


stats.pearsonr([weat_sos_eth_chan, weat_sos_lgtbq_chan,weat_sos_women_chan], crime_rate )


# ## Online Extremism and Online Hate

# In[13]:


hate_finland = [0.67, 0.63, 0.25]
hate_us = [0.60, 0.61, 0.44]
germany = [0.48, 0.50, 0.20]
uk = [0.57, 0.55, 0.44]


# In[14]:


print("w2v_finland",stats.pearsonr([weat_sos_eth_w2v, weat_sos_lgtbq_w2v, weat_sos_women_w2v], hate_finland))
print("glove_wk_finland",stats.pearsonr([weat_sos_eth_glove_wk, weat_sos_lgtbq_glove_wk, weat_sos_women_glove_wk], hate_finland))
print("glove_twitter",stats.pearsonr([weat_sos_eth_glove_twitter, weat_sos_lgtbq_glove_twitter, weat_sos_women_glove_twitter], hate_finland))
print("ud_finland",stats.pearsonr([weat_sos_eth_ud, weat_sos_lgtbq_ud, weat_sos_women_ud], hate_finland))
print("chan_finland",stats.pearsonr([weat_sos_eth_chan, weat_sos_lgtbq_chan, weat_sos_women_chan], hate_finland))


# In[15]:


print("w2v_us",stats.pearsonr([weat_sos_eth_w2v, weat_sos_lgtbq_w2v, weat_sos_women_w2v], hate_us))
print("glove_wk_us",stats.pearsonr([weat_sos_eth_glove_wk, weat_sos_lgtbq_glove_wk, weat_sos_women_glove_wk], hate_us))
print("glove_twitter_us",stats.pearsonr([weat_sos_eth_glove_twitter, weat_sos_lgtbq_glove_twitter, weat_sos_women_glove_twitter], hate_us))
print("ud_us",stats.pearsonr([weat_sos_eth_ud, weat_sos_lgtbq_ud, weat_sos_women_ud], hate_us))
print("chan_us",stats.pearsonr([weat_sos_eth_chan, weat_sos_lgtbq_chan, weat_sos_women_chan], hate_us))


# In[16]:


print("w2v_germany",stats.pearsonr([weat_sos_eth_w2v, weat_sos_lgtbq_w2v, weat_sos_women_w2v], germany))
print("glove_wk_germany",stats.pearsonr([weat_sos_eth_glove_wk, weat_sos_lgtbq_glove_wk, weat_sos_women_glove_wk], germany))
print("glove_twitter_germany",stats.pearsonr([weat_sos_eth_glove_twitter, weat_sos_lgtbq_glove_twitter, weat_sos_women_glove_twitter], germany))
print("ud_germany",stats.pearsonr([weat_sos_eth_ud, weat_sos_lgtbq_ud, weat_sos_women_ud], germany))
print("chan_germany",stats.pearsonr([weat_sos_eth_chan, weat_sos_lgtbq_chan, weat_sos_women_chan], germany))


# In[17]:


print("w2v_uk",stats.pearsonr([weat_sos_eth_w2v, weat_sos_lgtbq_w2v, weat_sos_women_w2v], uk))
print("glove_wk_uk",stats.pearsonr([weat_sos_eth_glove_wk, weat_sos_lgtbq_glove_wk, weat_sos_women_glove_wk], uk))
print("glove_twitter_uk",stats.pearsonr([weat_sos_eth_glove_twitter, weat_sos_lgtbq_glove_twitter, weat_sos_women_glove_twitter], uk))
print("ud_uk",stats.pearsonr([weat_sos_eth_ud, weat_sos_lgtbq_ud, weat_sos_women_ud], uk))
print("chan_uk",stats.pearsonr([weat_sos_eth_chan, weat_sos_lgtbq_chan, weat_sos_women_chan], uk))


# In[18]:


## difference between majoirty and minority added (online harassment) (https://www.womensmediacenter.com/speech-project/research-statistics)


# In[19]:


bullying_diff = [0.23,0.25,0.31]


# In[20]:


print("w2v_uk",stats.pearsonr([weat_sos_eth_w2v, weat_sos_lgtbq_w2v, weat_sos_women_w2v], bullying_diff))
print("glove_wk_uk",stats.pearsonr([weat_sos_eth_glove_wk, weat_sos_lgtbq_glove_wk, weat_sos_women_glove_wk], bullying_diff))
print("glove_twitter_uk",stats.pearsonr([weat_sos_eth_glove_twitter, weat_sos_lgtbq_glove_twitter, weat_sos_women_glove_twitter], bullying_diff))
print("ud_uk",stats.pearsonr([weat_sos_eth_ud, weat_sos_lgtbq_ud, weat_sos_women_ud], bullying_diff))
print("chan_uk",stats.pearsonr([weat_sos_eth_chan, weat_sos_lgtbq_chan, weat_sos_women_chan], bullying_diff))


# In[21]:


## https://enough.org/stats_cyberbullying


# In[22]:


bullying = [0.17,0.15,0.2]


# In[23]:


print("w2v_uk",stats.pearsonr([weat_sos_eth_w2v, weat_sos_lgtbq_w2v, weat_sos_women_w2v], bullying))
print("glove_wk_uk",stats.pearsonr([weat_sos_eth_glove_wk, weat_sos_lgtbq_glove_wk, weat_sos_women_glove_wk], bullying))
print("glove_twitter_uk",stats.pearsonr([weat_sos_eth_glove_twitter, weat_sos_lgtbq_glove_twitter, weat_sos_women_glove_twitter], bullying))
print("ud_uk",stats.pearsonr([weat_sos_eth_ud, weat_sos_lgtbq_ud, weat_sos_women_ud], bullying))
print("chan_uk",stats.pearsonr([weat_sos_eth_chan, weat_sos_lgtbq_chan, weat_sos_women_chan], bullying))


# In[ ]:




