import streamlit as st
from pandas.core.frame import DataFrame
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm_notebook as tqdm
import time
from random import shuffle
from nltk.corpus import wordnet
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import random
import json
import os

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

from gensim.test.utils import datapath



st.title("Word Embeddings for Business Entities")
option = st.sidebar.selectbox('Choose one from below',('Choose your own antonym pairs', 'Get top 5 antonym pairs based on the company ticker', 'PCA'))
st.write('You selected:', option)

polar_embedding_company = pd.read_csv('C:\\Users\\bhura\\Downloads\\POLAR-GloVeWiki-bus-antonyms-inter.csv')
fortune_company = pd.read_csv('C:\\Users\\bhura\\Downloads\\Fortune Global 500 companies.csv',encoding= 'unicode_escape')
polar_list_company_name = polar_embedding_company.iloc[:,0]

import pandas as pd

company = pd.read_csv('C:\\Users\\bhura\\Downloads\\International_Fortune_Google.csv')
name_list = company['0']
new_df= pd.read_csv('C:\\Users\\bhura\\Downloads\\POLAR-GoogleNews-bus-antonyms-inter.csv')

import plotly
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA

def display_pca_scatterplot_3D(model, user_input=None, words=None, label=None, color_map=None, topn=25, sample=10):

    word_vectors = word_vectors = np.array(model.loc[:,model.columns!='Unnamed: 0'])
    three_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:3]

    data = []
    count = 0
    for i in range (len(user_input)):

        trace = go.Scatter3d(
            x = three_dim[count:count+topn,0], 
            y = three_dim[count:count+topn,1],  
            z = three_dim[count:count+topn,2],
            text = words[count:count+topn],
            name = user_input[i],
            textposition = "top center",
            textfont_size = 20,
            mode = 'markers+text',
            marker = {
                'size': 10,
                'opacity': 0.8,
                'color': 2
            }

        )
        
                # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
                # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
    
        data.append(trace)
        count = count+topn

    trace_input = go.Scatter3d(
        x = three_dim[count:,0], 
        y = three_dim[count:,1],  
        z = three_dim[count:,2],
        text = words[count:],
        name = 'input words',
        textposition = "top center",
        textfont_size = 20,
        mode = 'markers+text',
        marker = {
            'size': 10,
            'opacity': 1,
            'color': 'black'
        }
        )

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
    data.append(trace_input)

    # Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )


    plot_figure = go.Figure(data = data, layout = layout)
    st.plotly_chart(plot_figure)

def display_pca_scatterplot_2D(model, user_input=None, words=None, label=None, color_map=None, topn=15, sample=10):

    word_vectors = word_vectors = np.array(model.loc[:,model.columns!='Unnamed: 0'])
    three_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]

    data = []
    count = 0
    
    for i in range (len(user_input)):

        trace = go.Scatter(
            x = three_dim[count:count+topn,0], 
            y = three_dim[count:count+topn,1],  
            text = words[count:count+topn],
            name = user_input[i],
            textposition = "top center",
            textfont_size = 20,
            mode = 'markers+text',
            marker = {
                'size': 10,
                'opacity': 0.8,
                'color': 2
            }

        )

        data.append(trace)
        count = count+topn

    trace_input = go.Scatter(
        x = three_dim[count:,0], 
        y = three_dim[count:,1],  
        text = words[count:],
        name = 'input words',
        textposition = "top center",
        textfont_size = 20,
        mode = 'markers+text',
        marker = {
            'size': 10,
            'opacity': 1,
            'color': 'black'
        }
        )
            
    data.append(trace_input)
    
# Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )


    plot_figure = go.Figure(data = data, layout = layout)
    st.plotly_chart(plot_figure)
        

Antonym_list = [('product', 'service'),
('essential', 'luxury'),
('technical', 'natural'),
('renewable', 'nonrenewable'),
('advertising', 'rumour'), 
('lease', 'sell'),
('demand', 'supply'),
('wfh', 'wfo'),
('child', 'childless'),
('remote', 'physical'),
('salary', 'goodies'),
('store', 'online'),
('details', 'outlines'),
('stakeholders', 'spectators'),
('isolating', 'social'),
('goal', 'task'),
('employees', 'consultant'),
('cost', 'revenue'),
('seasonal', 'temporary'),
('alliance', 'proprietorship'),
('loss', 'profit'),
('integrity', 'corruption'),
('international', 'local'),
('corporate', 'individual'),
('order', 'disorder'),
('solution', 'problem'),
('manager', 'worker'),
('diversity', 'uniformity'),
('public', 'private'),
('strategic', 'impulsive'),
('innovator', 'follower'),
('bankruptcy', 'prosperity'),
('growth', 'decline'),
('sustainable', 'unsustainable'),
('family', 'work'),
('criminal', 'rightful'),
('financial', 'artisanal'),
('supplier', 'purchaser'),
('commitment', 'rejection'),
('professional', 'amateur'),
('independent', 'dependent'),
('digital', 'analogue'),
('marketing', 'secret'),
('secure', 'risky'),
('longterm', 'shortterm'), 
('responsible', 'neglect'), 
('ethical', 'unethical'), 
('beneficial', 'harmful'), 
('diversity', 'uniformity'), 
('trust', 'mistrust'), 
('teamwork', 'individualism'), 
('opportunity', 'threat'), 
('innovative', 'traditional'), 
('flexible', 'rigid'), 
('ambiguity', 'clarity'), 
('feminine', 'masculine'), 
('globally', 'locally'), 
('insiders', 'outsiders'), 
('foreigners', 'natives'), 
('minorities', 'majority'),
('transparency', 'obscurity'),
('discrimination', 'impartial'),
('credible', 'deceptive'),
('environment', 'pollution'),
('pressure', 'relax'),
('growth', 'decline'),
('satisfied', 'unsatisfied'),
('diplomatic', 'undiplomatic'),
('motivate', 'demotivate'),
('communicative', 'uncommunicative'),
('connected', 'disconnected'),
('autonomous', 'micromanagement'),
('nurture', 'neglect'),
('progressive', 'conservative'),
('rewarding', 'unrewarding'),
('bias', 'unbias'),
('challenge', 'obscurity'),
('collaborated', 'siloed'),
('outdated', 'modern'),
('effortless', 'demanding'),
('economic', 'overpriced'),
('widespread', 'local'),
('freedom', 'captive'),
('consistent', 'inconsistent')]

if option == 'PCA':

    dimension = st.sidebar.selectbox(
     "Select the dimension of the visualization",
     ('2D', '3D'))

    if(dimension == "3D"):
        display_pca_scatterplot_3D(model=new_df[0:50], user_input=['Apple', 'Shell', 'Boeing'], words=name_list)
    if(dimension == "2D"):
        display_pca_scatterplot_2D(model=new_df[0:50], user_input=['Centene', 'Microsoft', 'Walmart'], words=name_list)

if option == 'Choose your own antonym pairs':
    dimension = st.sidebar.selectbox(
     "Select the dimension of the visualization",
     ('2D', '3D'))

    antonym_pair1 = st.sidebar.selectbox(
     "Select the Antonymn pair 1",
     Antonym_list)
    character_1 = '{}'.format(''.join(map(str, antonym_pair1)))
    character_1_list = polar_embedding_company[str(character_1)]

    antonym_pair2 = st.sidebar.selectbox(
     "Select the Antonymn pair 2",
     Antonym_list)
    character_2 = '{}'.format(''.join(map(str, antonym_pair2)))
    character_2_list = polar_embedding_company[str(character_2)]

    if(dimension == '3D'):
        antonym_pair3 = st.sidebar.selectbox(
     "Select the Antonymn pair 3",
     Antonym_list)
        character_3 = '{}'.format(''.join(map(str, antonym_pair3)))
        character_3_list = polar_embedding_company[str(character_3)]

    if(dimension == "2D"):
        company_location=[]
        fortune_company_name = []
        company_index=0
        counter=0
        for index, row in fortune_company.iterrows(): 
            s = row['Company']
            s = s.lower()
            s = s.replace(" ","")   
            if s == polar_list_company_name[company_index]:    
                company_index = company_index+1
                company_location.append(row['Location'])
                fortune_company_name.append(row['Company'])
                counter=counter+1

  
        data = {
                'Company Name' : fortune_company_name,    
                'Location': company_location,      
                character_1 : character_1_list,
                character_2 : character_2_list}

        df = pd.DataFrame(data)
        fig = px.scatter(df, x=character_1, y=character_2, text="Company Name", size_max=1,color='Location', color_discrete_sequence= px.colors.qualitative.Alphabet)
        fig.update_traces(textposition='top center')
        fig.update_layout(title_text='Companies', title_x=0.5)

        st.plotly_chart(fig)

    else:
        company_location=[]
        fortune_company_name = []
        company_index=0
        counter=0
        for index, row in fortune_company.iterrows(): 
            s = row['Company']
            s = s.lower()
            s = s.replace(" ","")   
            if s == polar_list_company_name[company_index]:    
                company_index = company_index+1
                company_location.append(row['Location'])
                fortune_company_name.append(row['Company'])
                counter=counter+1

  
        data = {
                'Company Name' : fortune_company_name,    
                'Location': company_location,      
                character_1 : character_1_list,
                character_2 : character_2_list,
                character_3 : character_3_list}

        df = pd.DataFrame(data)
        fig = px.scatter_3d(df, x=character_1, y=character_2, z=character_3, text="Company Name", size_max=1,color='Location', color_discrete_sequence= px.colors.qualitative.Alphabet)
        fig.update_traces(textposition='top center')
        fig.update_layout(title_text='Companies', title_x=0.5)
        st.plotly_chart(fig)

if option == 'Get top 5 antonym pairs based on the company ticker':
    st.write("Please enter a ticker of a company")
    ticker = st.text_input("ticker/Name")

    submit = st.button("Enter")

    if submit:
        st.write("You submitted the request")

from PIL import Image

# tickers = df.columns
# tickers = tickers[2:]




def load_image(image_file):
	img = Image.open(image_file)
	return img

# if ticker in tickers:
#     df = df.loc[df['tickers'] == ticker]
#     df = df[df.columns.to_series().sample(5)]
#     st.write(df)
#     fig = df.plot(kind='barh')
#     fig.figure.savefig('file.png')
#     st.image(load_image('C:\\Users\\bhura\\Downloads\\file.png'))