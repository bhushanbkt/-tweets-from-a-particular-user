#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

warnings.simplefilter(action='ignore', category=FutureWarning)


# **Data cleaning**

# In[2]:



df=pd.read_csv('training.1600000.processed.noemoticon.csv',encoding='latin-1')
df


# In[3]:


df.info()


# In[4]:


df.rename(columns={'Mon Apr 06 22:19:45 PDT 2009':'Date'},inplace=True)
    


# In[5]:


df.rename(columns={"@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D":"Tweet"},inplace=True)
df.head()


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.drop(columns='NO_QUERY',axis=1,inplace=True)
df.head()


# In[9]:


df['0'].unique()


# In[10]:


df.drop(columns='0',axis=1,inplace=True)
df.head()


# In[11]:


df.tail()


# In[12]:


df.drop('1467810369',axis=1,inplace=True)


# In[13]:


df.duplicated().sum()


# In[14]:


#remove dublicate
df=df.drop_duplicates(keep='first')


# In[15]:


df.duplicated().sum()


# In[16]:


df


# In[17]:


df.rename(columns={'_TheSpecialOne_':'the_special_one'},inplace=True)


# In[18]:


df.info()


# In[19]:


get_ipython().system(' pip install nltk')


# In[20]:


import nltk


# In[21]:


nltk.download('punkt')


# In[22]:


df['Tweet'][0]


# In[23]:


df['num_of_characters']=df['Tweet'].apply(len)


# In[24]:


df


# 
# ## Data Preprocessing
# 
# *lower case**
# 
# *Tokenization**
# 
# *Removing special characters**
# 
# *Steaming**

# In[56]:


import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[58]:


def text_proce(text):
    text = text.lower()
    text= nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


# In[47]:


import string
string.punctuation


# In[46]:


stopwords.words('english')


# In[ ]:





# In[ ]:


df['Transformed_tweet']=df['Tweet'].apply(text_proce)


# In[ ]:


df


# In[ ]:




