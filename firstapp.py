#!/usr/bin/env python
# coding: utf-8

# In[8]:


import streamlit as st
import numpy as np
import pandas as pd


# In[9]:


st.title('First app')


# In[10]:


st.write("Here's our first attempt at using data to create a table:")
df = pd.read_csv('dataset-of-10s.csv')
df


# In[11]:


chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)


# In[12]:


map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)


# In[13]:


if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data


# In[14]:

artistlist = df.loc[0:9]['artist'].tolist()


option = st.selectbox(
    'Which artist do you like best?',
     artistlist)

st.write('You selected:', option)


# In[15]:


left_column, right_column = st.columns(2)
pressed = left_column.button('Press me?')
if pressed:
  right_column.write("Woohoo!")

expander = st.expander("FAQ")
expander.write("Here you could put in some really, really long explanations...")


# In[16]:


import time
'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'


# In[ ]:




