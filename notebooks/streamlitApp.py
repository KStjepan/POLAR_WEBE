from distutils import core
from operator import index
import streamlit as st
from pandas.core.frame import DataFrame
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg
import time
from random import shuffle
import matplotlib.pyplot as plt
import sys
import nltk 
from nltk.corpus import wordnet 
import gc
from collections import defaultdict
import random
import json
import os
import plotly.graph_objs as go
# import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from gsheetsdb import connect
import plotly
import colorlover as cl
import plotly.offline as py
import seaborn as sns 
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error
import plotly.express as px

def polar_list(list):
  
  right_polar_list = []
  left_polar_list = []
  for i in range(0,len(list)):
    
    left_polar_list.append(list[i][0].replace("-","_"))
    right_polar_list.append(list[i][1].replace("-","_"))

  return left_polar_list,right_polar_list

def alphabetical_list_creation(list):
  new_list = []
  
  for i in range(0,len(list)):
    index_0 = list[i][0].replace("-","_")
    index_1 = list[i][1].replace("-","_")
    
    if index_0 < index_1:
      val = index_0+"-"+index_1
      new_list.append(val)
      
    else:
      val = index_1+"-"+index_0
      new_list.append(val)
      
  return new_list

def company_count(company_df,input_list,polar_embedding):  

  # we then find the number of companies grouped on the basis of location
  for i in input_list:       
    j = i.replace("-","")    
    j = j.replace("_","-")
    
    subset_df2 = polar_embedding[polar_embedding[j] < 0]
    company_inclined_to_left_polar_df1 = subset_df2['Location'].value_counts()
    left_polar = i.split("-")[0]
    
    company_inclined_to_left_polar_df1 = pd.DataFrame({'Country':company_inclined_to_left_polar_df1.index, left_polar :company_inclined_to_left_polar_df1.values})
    company_df=pd.merge(company_df, company_inclined_to_left_polar_df1, how='left',on='Country')    
    company_df[left_polar] = round( company_df[left_polar] / company_df.iloc[:,1] * 100)

    subset_df1 = polar_embedding[polar_embedding[j] > 0]
    company_inclined_to_right_polar_df1 = subset_df1['Location'].value_counts()
    right_polar = i.split("-")[1]
    
    company_inclined_to_right_polar_df1 = pd.DataFrame({'Country':company_inclined_to_right_polar_df1.index, right_polar :company_inclined_to_right_polar_df1.values})
    company_df=pd.merge(company_df, company_inclined_to_right_polar_df1, how='left',on='Country')    
    company_df[right_polar] = round( company_df[right_polar] / company_df.iloc[:,1] * 100)


  company_df = company_df.fillna(0)

  # We are considering only the countries if the numberof companies in the country is over 3
  company_df = company_df[company_df['Total Count'] > 3]

  return company_df

def polar_ranking(polar_list,total_score,ranking,company_df):
  total_sum=0
  total_sum_list=[]
  polar_ranking_list = []
  polar_index=0
  for index,row in company_df.iterrows():  
    
    for i in polar_list:
      
      total_sum = total_sum + (row[i])
    #print(company_df.iloc[index,2:])  
    total_sum_list.append(total_sum/len(polar_list))
    polar_ranking_list.append(index+1)
    total_sum = 0

  company_df[total_score] = total_sum_list
  company_df= company_df.sort_values(by=[total_score],ascending=False)
  company_df[ranking] = polar_ranking_list

  return company_df


def mean_absolute_error_score(merged_df,dimension):
  MAE_of_Score = []
  MAE_of_Score.append(mean_absolute_error(merged_df[dimension], merged_df["Total Score Random"]))
  MAE_of_Score.append(mean_absolute_error(merged_df[dimension], merged_df["Total Score Nearest Random"]))
  MAE_of_Score.append(mean_absolute_error(merged_df[dimension], merged_df["Total Score Human"]))
  MAE_of_Score.append(mean_absolute_error(merged_df[dimension], merged_df["Total Score Nearest Human"]))
  return MAE_of_Score

def mean_absolute_error_rank(merged_df,dimension_ranking):
  MAE = []
  MAE.append(mean_absolute_error(merged_df[dimension_ranking], merged_df["Polar Rank R"]))
  MAE.append(mean_absolute_error(merged_df[dimension_ranking], merged_df["Polar Rank Nearest R"]))
  MAE.append(mean_absolute_error(merged_df[dimension_ranking], merged_df["Polar Rank H"]))
  MAE.append(mean_absolute_error(merged_df[dimension_ranking], merged_df["Polar Rank Nearest H"]))
  return MAE

def correlation_calc(merged_df,dimension_ranking):
  correlation = []
  correlation.append(merged_df["Polar Rank R"].corr(merged_df[dimension_ranking]))
  correlation.append(merged_df["Polar Rank Nearest R"].corr(merged_df[dimension_ranking]))
  correlation.append(merged_df["Polar Rank H"].corr(merged_df[dimension_ranking]))
  correlation.append(merged_df["Polar Rank Nearest H"].corr(merged_df[dimension_ranking]))
  return correlation

# Power Distance
list_powerdistance_random =[('make', 'break'), ('cameraman', 'playwright'), ('mystical', 'factual'), ('promotional', 'defamation'), ('iconic', 'unknown')]
nearest_random_list_powerdistance =[('making', 'breaking'), ('cameramen', 'dramatist'), ('magical', 'inaccuracies'), ('promo', 'libel'), ('recognizable', 'undetermined')]
list_powerdistance =[('hierarchical','nonhierarchical'),('superior','equal'),('leader','subordinate'),('inequality','equality'),('autocrat','democrat')]
nearest_human_list_powerdistance = [('hierarchy', 'consensusbased'), ('inferior', 'equalitys'), ('leaders', 'subordinates'), ('inequalities', 'equals'), ('autocratic', 'senator')]

# Individualism
list_individualism_random = [('lop', 'secure'), ('shah', 'poor'), ('pneumatic', 'solid'), ('interpret', 'misinterpret'), ('confer', 'refuse')]
nearest_random_list_individualism= [('buri', 'securing'), ('ahmad', 'poorer'), ('hydraulic', 'consistent'), ('interpreting', 'misunderstand'), ('conferring', 'refusing')]
list_individualism = [('individuality','community'),('selfinterest','harmony'),('tasks','relationships'),('individual','groups'),('universalism','particularism')]
nearest_human_list_individualism = [('originality', 'communities'), ('selfishness', 'harmonious'), ('task', 'relationship'), ('individuals', 'group'), ('mangxamba', 'unitarianism')]

# Masculinity
list_masculinity_random = [('try', 'abstain'), ('fatalistic', 'freewill'), ('knowledgeable', 'uninformed'), ('confine', 'free'), ('fan', 'warm')]
nearest_random_list_masculinity = [('trying', 'abstaining'), ('nonchalant', 'gmv'), ('knowledgable', 'misinformed'), ('confining', 'freedom'), ('fans', 'cool')]
list_masculinity = [('achievement', 'support'),('competitive', 'caring'),('assertive', 'submissive'),('ambitious', 'unambitious'),('sucess','cooperation')]
nearest_human_list_masculinity = [('achievements', 'supported'), ('competition', 'loving'), ('forceful', 'subservient'), ('undertaking', 'unathletic'), ('ufauthor', 'bilateral')]

# long term Orientation
list_longterm_random = [('innovator', 'follower'), ('sensory', 'numb'), ('hedge', 'squander'), ('arachnid', 'serpent'), ('disclose', 'secrete')]
nearest_random_list_longterm = [('visionary', 'disciple'), ('auditory', 'numbed'), ('fund', 'squandering'), ('itsy', 'serpents'), ('disclosing', 'secreted')]
list_longterm = [('pragmatic','normative'),('progress','preserve'),('adapt','conserve'),('developing','stable'),('advance','retain')]
nearest_human_list_longterm = [('pragmatism', 'conceptions'), ('efforts', 'preserving'), ('adapting', 'conserving'), ('develop', 'stability'), ('advancing', 'retained')]

# Indulgence
list_indulgence_random = [('diagnose', 'sicken'), ('intercourse', 'disconnection'), ('sensory', 'sensorial'), ('emasculate', 'strengthen'), ('metropolitan', 'rural')]
nearest_random_list_indulgence = [('diagnosing', 'sickens'), ('sexual', 'disconnect'), ('auditory', 'skorokhod'), ('disempower', 'strengthening'), ('metro', 'urban')]
list_indulgence = [('fulfillment','restriction'),('satisfaction','limitation'),('liberty','moderation'),('expand','direct'),('freedom','regulation')]
nearest_human_list_indulgence = [('fulfilment', 'restrictions'), ('satisfied', 'limitations'), ('fredom', 'restraint'), ('expanding', 'indirect'), ('freedoms', 'regulations')]

# Unceratinity Avoidance
list_uncertainity_avoidance_random = [('stretcher', 'compressor'), ('amalgamate', 'separate'), ('caretaker', 'assailant'), ('taker', 'violator'), ('contaminate', 'sterilize')]
nearest_random_list_uncertainity_avoidance = [('stretchers', 'compressors'), ('amalgamating', 'separately'), ('interim', 'assailants'), ('takers', 'violators'), ('contaminating', 'sterilized')]
list_uncertainity_avoidance = [('clarity','complexity'),('clear','ambiguous'),('certain','uncertain'),('uniformity','diversity'),('agreement','variation')]
nearest_human_list_uncertainity_avoidance = [('simplicity', 'complexities'), ('yet', 'vague'), ('particular', 'unclear'), ('homogeneity', 'diverse'), ('agreements', 'variations')]


def display_pca_scatterplot_2D(model):

    data = []
    for i in range(-1,model['cluster'].max()+1):
        
        word_vectors = model.loc[model['cluster']==i]
        scat_text = word_vectors['Unnamed: 0']
        two_dim = word_vectors[['two_dim1','two_dim2']].to_numpy()
        trace = go.Scatter(
            x = two_dim[:,0], 
            y = two_dim[:,1],  
            text = scat_text[:],
            name = 'Cluster'+str(i+2),
            textposition = "top center",
            textfont_size = 10,#20
            mode = 'markers+text',
            marker = {
                'size': 10,#10
                'opacity': 0.8,
                'color': i
            }
        )      
        data.append(trace)

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
        width = 1000,#1200
        height = 500 #700
        )

    plot_figure = go.Figure(data = data, layout = layout)
    st.plotly_chart(plot_figure)

def display_pca_scatterplot_3D(model):

    data = []
    for i in range(-1,model['cluster'].max()+1):
        
        word_vectors = model.loc[model['cluster']==i]
        scat_text = word_vectors['Unnamed: 0']
        three_dim = word_vectors[['three_dim1','three_dim2','three_dim3']].to_numpy()
        trace = go.Scatter3d(
            x = three_dim[:,0], 
            y = three_dim[:,1],
            z = three_dim[:,2],
            text = scat_text[:],
            name = 'Cluster'+str(i+2),
            textposition = "top center",
            textfont_size = 10,#20
            mode = 'markers+text',
            marker = {
                'size': 10,#10
                'opacity': 0.8,
                'color': i
            }
        )      
        data.append(trace)

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
        height = 700
        )

    plot_figure = go.Figure(data = data, layout = layout)
    st.plotly_chart(plot_figure)

Antonym_list = ['commitment rejection', 'manager worker', 'feminine masculine', 'globally locally',
                'family work', 'stakeholders spectators', 'discrimination impartial', 'challenge obscurity',
                'seasonal temporary',
                'alliance proprietorship', 'public private', 'details outlines', 'responsible neglect',
                'marketing secret', 'trust mistrust', 'independent dependent', 'integrity corruption',
                'bankruptcy prosperity', 'insiders outsiders', 'teamwork individualism', 'foreigners natives',
                'criminal rightful', 'strategic impulsive', 'environment pollution', 'diversity uniformity',
                'progressive conservative', 'salary goodies', 'innovator follower', 'pressure relax',
                'secure risky', 'remote physical', 'sustainable unsustainable', 'product service',
                'essential luxury', 'digital analogue', 'effortless demanding', 'nurture neglect',
                'professional amateur', 'ambiguity clarity', 'credible deceptive', 'widespread local',
                'freedom captive', 'order disorder',
                'goal task', 'cost revenue', 'demand supply', 'opportunity threat', 'flexible rigid',
                'isolating social', 'international local', 'innovative traditional', 'satisfied unsatisfied',
                'solution problem', 'store online', 'loss profit', 'ethical unethical',
                'beneficial harmful', 'economic overpriced', 'outdated modern', 'transparency obscurity',
                'lease sell', 'technical natural', 'consistent inconsistent', 'growth decline',
                'tangible intangible', 'employees consultant', 'financial artisanal', 'child childless',
                'connected disconnected', 'corporate individual']

st.title("Word Embeddings for Business Entities")
# option = st.sidebar.selectbox('Create your own polar pairs?',('Yes',  'No'))

check = st.sidebar.selectbox('Check for', ('Bias', 'Hofstede', 'PCA'))
 
if (check == 'Bias'):
    # CSS to inject contained in a string
    hide_dataframe_row_index = """
                <style>
                .row_heading.level0 {display:none}
                .blank {display:none}
                </style>
                """

    # Inject CSS with Markdown
    st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

    company_or_country = st.sidebar.selectbox('Check for', ('Companies', 'Countries', 'P-value'))
    if (company_or_country == 'Countries'):
        antonym_pair = st.sidebar.selectbox("Select the Antonymn pair", Antonym_list)

        antonym_pair = str(antonym_pair.replace(" ", "_"))

        gnews_url = "https://docs.google.com/spreadsheets/d/15I-OwV1vV6lB2SMJgKIS_auPfj0o7nbpOau6geqET2Q/edit?usp=sharing"
        wiki_url = "https://docs.google.com/spreadsheets/d/1zhgZFDLcci0DzCBPJMEY1nQP107536ReBCRZKjXYGxw/edit?usp=sharing"
        twitter_url = "https://docs.google.com/spreadsheets/d/19s5djluHPiKIk-2I0JSm019gz48cGVqKYzC8Q0dfeGY/edit?usp=sharing"
        reddit_url = "https://docs.google.com/spreadsheets/d/1l4gwQujj6C-EgF9ci7TYdPwNqe5Ef_HXxrj7TuretxM/edit?usp=sharing"

        conn = connect()
        gnews_rows = conn.execute(f'SELECT * FROM "{gnews_url}"')
        wiki_rows = conn.execute(f'SELECT * FROM "{wiki_url}"')
        twitter_rows = conn.execute(f'SELECT * FROM "{twitter_url}"')
        reddit_rows = conn.execute(f'SELECT * FROM "{reddit_url}"')

        gnews = pd.DataFrame(gnews_rows)
        country = list(gnews['Country'].str.split('_').str[0])
        gnews.set_index('Country', inplace=True)

        wiki = pd.DataFrame(wiki_rows)
        wiki.set_index('Country', inplace=True)

        twitter = pd.DataFrame(twitter_rows)
        twitter.set_index('Country', inplace=True)

        reddit = pd.DataFrame(reddit_rows)
        reddit.set_index('Country', inplace=True)

        country = st.sidebar.multiselect('Select Upto 5 countries', country)

        country_gnews = [i + "_gnews" for i in country]
        country_gnews = gnews.loc[country_gnews]

        country_wiki = [i + "_wiki" for i in country]
        country_wiki = wiki.loc[country_wiki]

        country_reddit = [i + "_reddit" for i in country]
        country_reddit = reddit.loc[country_reddit]

        country_twitter = [i + "_twitter" for i in country]
        country_twitter = twitter.loc[country_twitter]

        trace0 = go.Scatter(
            {
                'x': country_reddit[antonym_pair],
                'y': country,
                'legendgroup': 'Reddit',
                'name': 'Reddit',
                'mode': 'markers',
                'marker': {
                    'color': cl.scales['9']['div']['Spectral'][0],
                    'size': 40,
                }
                # 'text': reddit['Country']
            })

        trace1 = go.Scatter(
            {
                'x': country_wiki[antonym_pair],
                'y': country,
                'legendgroup': 'Wikipedia',
                'name': 'Wikipedia',
                'mode': 'markers',
                'marker': {
                    'color': cl.scales['9']['div']['Spectral'][2],
                    'size': 40
                }
                # 'text': wiki['Country']
            })

        trace2 = go.Scatter(
            {
                'x': country_twitter[antonym_pair],
                'y': country,
                'legendgroup': 'Twitter',
                'name': 'Twitter',
                'mode': 'markers',
                'marker': {
                    'color': cl.scales['9']['div']['Spectral'][8],
                    'size': 40
                }
                # 'text': twitter['Country']
            })

        trace3 = go.Scatter(
            {
                'x': country_gnews[antonym_pair],
                'y': country,
                'legendgroup': 'Google News',
                'name': 'Google News',
                'mode': 'markers',
                'marker': {
                    'color': cl.scales['9']['div']['Spectral'][6],
                    'size': 40
                }
                # 'text': gnews['Country']
            })

        layout = go.Layout(
            title='Business Entities',
            hovermode='closest',
            xaxis=dict(
                title=antonym_pair
            ),
            yaxis=dict(
                title='Companies'
            ),
            showlegend=True,
            # CENTER = 0
        )

        fig = go.Figure(data=[trace0, trace1, trace2, trace3], layout=layout)
        st.plotly_chart(fig)

    elif (company_or_country == 'Companies'):

        antonym_pair = st.sidebar.selectbox("Select the Antonymn pair", Antonym_list)

        antonym_pair = str(antonym_pair.replace(" ", "_"))

        gnews_url = "https://docs.google.com/spreadsheets/d/1coqpsqHM2LxP0H3Xg89QmJgulW3gy98gKw6EnLP65oo/edit?usp=sharing"
        wiki_url = "https://docs.google.com/spreadsheets/d/17wkBZudbjD94dJ5tGH65Oz-sr0yeO8Y9NKYW4bZM7Ok/edit?usp=sharing"
        twitter_url = "https://docs.google.com/spreadsheets/d/1eNuZJXiSDQXGoax5ls_qFOuWHhUBoLwZH2vAn3xQtt4/edit?usp=sharing"
        reddit_url = "https://docs.google.com/spreadsheets/d/17hxKvAxzrrSWfxplsWLygO4D1flYdAciXKq4NKLUZHc/edit?usp=sharing"

        conn = connect()
        gnews_rows = conn.execute(f'SELECT * FROM "{gnews_url}"')
        wiki_rows = conn.execute(f'SELECT * FROM "{wiki_url}"')
        twitter_rows = conn.execute(f'SELECT * FROM "{twitter_url}"')
        reddit_rows = conn.execute(f'SELECT * FROM "{reddit_url}"')

        gnews = pd.DataFrame(gnews_rows)
        company = list(set(gnews['Company_Name']))
        gnews.set_index('Company_Name', inplace=True)

        wiki = pd.DataFrame(wiki_rows)
        wiki.set_index('Company_Name', inplace=True)

        twitter = pd.DataFrame(twitter_rows)
        twitter.set_index('Company_Name', inplace=True)

        reddit = pd.DataFrame(reddit_rows)
        reddit.set_index('Company_Name', inplace=True)

        company = st.sidebar.multiselect('Select Upto 5 companies', company)

        # country_gnews = [i+"_gnews" for i in country]
        company_gnews = gnews.loc[company]

        # country_wiki = [i+"_wiki" for i in country]
        company_wiki = wiki.loc[company]

        # country_reddit = [i+"_reddit" for i in country]
        company_reddit = reddit.loc[company]

        # country_twitter = [i+"_twitter" for i in country]
        company_twitter = twitter.loc[company]

        # reddit = reddit.head(5)
        # wiki = wiki.head(5)
        # twitter = twitter.head(5)
        # gnews = gnews.head(5)

        trace0 = go.Scatter(
            {
                'x': company_reddit[antonym_pair],
                'y': company,
                'legendgroup': 'Reddit',
                'name': 'Reddit',
                'mode': 'markers',
                'marker': {
                    'color': cl.scales['9']['div']['Spectral'][0],
                    'size': 40,
                },
                # 'text': reddit['Country']
            })

        trace1 = go.Scatter(
            {
                'x': company_wiki[antonym_pair],
                'y': company,
                'legendgroup': 'Wikipedia',
                'name': 'Wikipedia',
                'mode': 'markers',
                'marker': {
                    'color': cl.scales['9']['div']['Spectral'][2],
                    'size': 40
                },
                # 'text': wiki['Country']
            })

        trace2 = go.Scatter(
            {
                'x': company_twitter[antonym_pair],
                'y': company,
                'legendgroup': 'Twitter',
                'name': 'Twitter',
                'mode': 'markers',
                'marker': {
                    'color': cl.scales['9']['div']['Spectral'][8],
                    'size': 40
                },
                # 'text': twitter['Country']
            })

        trace3 = go.Scatter(
            {
                'x': company_gnews[antonym_pair],
                'y': company,
                'legendgroup': 'Google News',
                'name': 'Google News',
                'mode': 'markers',
                'marker': {
                    'color': cl.scales['9']['div']['Spectral'][6],
                    'size': 40
                },
                # 'text': gnews['Country']
            })

        layout = go.Layout(
            title='Business Entities',
            hovermode='closest',
            xaxis=dict(
                title=antonym_pair
            ),
            yaxis=dict(
                title='Companies'
            ),
            showlegend=True,
            # CENTER = 0
        )

        fig = go.Figure(data=[trace0, trace1, trace2, trace3], layout=layout)
        st.plotly_chart(fig)

    elif(company_or_country == 'P-value'):
        test = st.sidebar.radio("Check T-test on",('pre-trained models', 'U.S and Non-U.S companies'))

        if(test == 'pre-trained models'):
            pvalue_url = "https://docs.google.com/spreadsheets/d/18HUCezv01mrWT0ntEF076ULbQoLw_FaZfd4qQyDXwEw/edit?usp=sharing"

            conn = connect()
            pvalue = conn.execute(f'SELECT * FROM "{pvalue_url}"')
            pvalue = pd.DataFrame(pvalue)

            test1 = st.sidebar.radio("Pre-trained Models",('Reddit & Wikipedia', 'Reddit & Twitter', 'Reddit & Google News', 'Google News & Twitter', 'Google News & Wikipedia', 'Twitter & Wikipedia'))
            
            if(test1 == 'Reddit & Wikipedia'):
                data = pvalue.iloc[:,0:2]
                data = data.dropna()
                st.write(data)

            elif(test1 == 'Reddit & Twitter'):
                data = pvalue.iloc[:,[0,2]]
                data = data.dropna()
                st.write(data)
            
            elif(test1 == 'Reddit & Google News'):
                data = pvalue.iloc[:,[0,3]]
                data = data.dropna()
                st.write(data)  

            elif(test1 == 'Google News & Twitter'):
                data = pvalue.iloc[:,[0,4]]
                data = data.dropna()
                st.write(data)  

            elif(test1 == 'Google News & Wikipedia'):
                data = pvalue.iloc[:,[0,5]]
                data = data.dropna()
                st.write(data)   

            elif(test1 == 'Twitter & Wikipedia'):
                data = pvalue.iloc[:,[0,6]]
                data = data.dropna()
                st.write(data)  

        elif(test == 'U.S and Non-U.S companies'):
            pvalue_url = "https://docs.google.com/spreadsheets/d/1sTOikbKuelgIFewL2SYaOgaXcLG7W-UCJ-jrcHDbQYU/edit?usp=sharing"

            conn = connect()
            pvalue = conn.execute(f'SELECT * FROM "{pvalue_url}"')
            pvalue = pd.DataFrame(pvalue)

            test1 = st.sidebar.radio("Pre-trained Models",('Reddit', 'Twitter', 'Google News',  'Wikipedia'))

            if(test1 == 'Reddit'):
                data = pvalue.iloc[:,0:2]
                data = data.dropna()
                st.dataframe(data)

            elif(test1 == 'Twitter'):
                data = pvalue.iloc[:,[0,2]]
                data = data.dropna()
                st.write(data)

            elif(test1 == 'Google News'):
                data = pvalue.iloc[:,[0,3]]
                data = data.dropna()
                st.write(data)

            elif(test1 == 'Wikipedia'):
                data = pvalue.iloc[:,[0,4]]
                data = data.dropna()
                st.write(data)

if (check == 'Hofstede'):
    Hofstede_dimensions = st.sidebar.selectbox('Check for', ('Power Distance', 'Individualism vs Collectivism','Masculinity vs Femininity', 'Long Term vs Short Term Orientation','Indulgence vs Restraint','Uncertainty Avoidance'))

    new_df_url = "https://docs.google.com/spreadsheets/d/1CzCINusz2boi7ziroOT0jlQnzvXWlMxs0x6Yv8hSzA8/edit?usp=sharing"
    fortune_500_company_url = "https://docs.google.com/spreadsheets/d/1sATMYArLD6e6tggHjAFlifkojVqssRRM4UvjI8z1AGc/edit?usp=sharing"
    hofstede_df_url = "https://docs.google.com/spreadsheets/d/1JLvLrAJh5kZKSKc65oEd6Rrnv-Da95Cg/edit?usp=sharing&ouid=118230191438546225615&rtpof=true&sd=true"

    conn = connect()
    new_df = conn.execute(f'SELECT * FROM "{new_df_url}"')
    new_df = pd.DataFrame(new_df)
    fortune_500_company = conn.execute(f'SELECT * FROM "{fortune_500_company_url}"')
    fortune_500_company = pd.DataFrame(fortune_500_company)
    hofstede_df = conn.execute(f'SELECT * FROM "{hofstede_df_url}"')
    hofstede_df = pd.DataFrame(hofstede_df)

    fortune_500_company['Company'] = fortune_500_company['Company'].str.lower()
    fortune_500_company['Company'] = fortune_500_company['Company'].str.replace(" ", "")

    polar_embedding = pd.merge(fortune_500_company, new_df, how="right", left_on="Company", right_on="company")

    polar_embedding = polar_embedding.drop(['Rank'], axis=1)  # This will drop the column Rank
    # st.write(polar_embedding)
    # polar_embedding = polar_embedding.drop(['Unnamed: 0'], axis=1)  # This will drop the column Rank

    # This will find the total number of companies in our data frame based on Location
    total_company_list_based_on_loc = polar_embedding['Location'].value_counts()
    total_company_count_df = pd.DataFrame({'Country': total_company_list_based_on_loc.index, 'Total Count': total_company_list_based_on_loc.values})

    hofstede_df=hofstede_df[hofstede_df.iloc[:,:]!="<NA>" ]  

    dim_index = ""
    dim_ranking = ""
    if (Hofstede_dimensions == 'Power Distance'):
      dim_index="Power_distance_index"
      dim_ranking="Power_distance_Ranking"
      
      left_polar_list_random,right_polar_list_random = polar_list(list_powerdistance_random)
      left_polar_list_nearest_random,right_polar_list_nearest_random = polar_list(nearest_random_list_powerdistance)
      left_polar_list_human,right_polar_list_human = polar_list(list_powerdistance)
      left_polar_list_nearest_human,right_polar_list_nearest_human = polar_list(nearest_human_list_powerdistance)


      input_list_random = alphabetical_list_creation(list_powerdistance_random)
      input_list_nearest_random = alphabetical_list_creation(nearest_random_list_powerdistance)
      input_list_human = alphabetical_list_creation(list_powerdistance)
      input_list_nearest_human = alphabetical_list_creation(nearest_human_list_powerdistance)

    elif (Hofstede_dimensions == 'Individualism vs Collectivism'):

      dim_index="Individualism_index"
      dim_ranking="Individualism_Ranking"
        
      left_polar_list_random,right_polar_list_random = polar_list(list_individualism_random)
      left_polar_list_nearest_random,right_polar_list_nearest_random = polar_list(nearest_random_list_individualism)
      left_polar_list_human,right_polar_list_human = polar_list(list_individualism)
      left_polar_list_nearest_human,right_polar_list_nearest_human = polar_list(nearest_human_list_individualism)

      input_list_random = alphabetical_list_creation(list_individualism_random)
      input_list_nearest_random = alphabetical_list_creation(nearest_random_list_individualism)
      input_list_human = alphabetical_list_creation(list_individualism)
      input_list_nearest_human = alphabetical_list_creation(nearest_human_list_individualism)

    elif (Hofstede_dimensions == 'Masculinity vs Femininity'):
      dim_index="Masculinity_index"
      dim_ranking="Masculinity_Ranking"
      
      left_polar_list_random,right_polar_list_random = polar_list(list_masculinity_random)
      left_polar_list_nearest_random,right_polar_list_nearest_random = polar_list(nearest_random_list_masculinity)
      left_polar_list_human,right_polar_list_human = polar_list(list_masculinity)
      left_polar_list_nearest_human,right_polar_list_nearest_human = polar_list(nearest_human_list_masculinity)


      input_list_random = alphabetical_list_creation(list_masculinity_random)
      input_list_nearest_random = alphabetical_list_creation(nearest_random_list_masculinity)
      input_list_human = alphabetical_list_creation(list_masculinity)
      input_list_nearest_human = alphabetical_list_creation(nearest_human_list_masculinity)
        
    elif (Hofstede_dimensions == 'Long Term vs Short Term Orientation'):
      dim_index="Long_term_orientation_index"
      dim_ranking="Long_term_orientation_Ranking"

      left_polar_list_random,right_polar_list_random = polar_list(list_longterm_random)
      left_polar_list_nearest_random,right_polar_list_nearest_random = polar_list(nearest_random_list_longterm)
      left_polar_list_human,right_polar_list_human = polar_list(list_longterm)
      left_polar_list_nearest_human,right_polar_list_nearest_human = polar_list(nearest_human_list_longterm)

      input_list_random = alphabetical_list_creation(list_longterm_random)
      input_list_nearest_random = alphabetical_list_creation(nearest_random_list_longterm)
      input_list_human = alphabetical_list_creation(list_longterm)
      input_list_nearest_human = alphabetical_list_creation(nearest_human_list_longterm)
        
    elif (Hofstede_dimensions == 'Indulgence vs Restraint'):
      dim_index="Indulgence_index"
      dim_ranking="Indulgence_Ranking"

      left_polar_list_random,right_polar_list_random = polar_list(list_indulgence_random)
      left_polar_list_nearest_random,right_polar_list_nearest_random = polar_list(nearest_random_list_indulgence)
      left_polar_list_human,right_polar_list_human = polar_list(list_indulgence)
      left_polar_list_nearest_human,right_polar_list_nearest_human = polar_list(nearest_human_list_indulgence)

      input_list_random = alphabetical_list_creation(list_indulgence_random)
      input_list_nearest_random = alphabetical_list_creation(nearest_random_list_indulgence)
      input_list_human = alphabetical_list_creation(list_indulgence)
      input_list_nearest_human = alphabetical_list_creation(nearest_human_list_indulgence)
        
    elif (Hofstede_dimensions == 'Uncertainty Avoidance'):
      dim_index="Uncertainty_avoidance_index"
      dim_ranking="Uncertainty_avoidance_Ranking"
      
      left_polar_list_random,right_polar_list_random = polar_list(list_uncertainity_avoidance_random)
      left_polar_list_nearest_random,right_polar_list_nearest_random = polar_list(nearest_random_list_uncertainity_avoidance)
      left_polar_list_human,right_polar_list_human = polar_list(list_uncertainity_avoidance)
      left_polar_list_nearest_human,right_polar_list_nearest_human = polar_list(nearest_human_list_uncertainity_avoidance)

      input_list_random = alphabetical_list_creation(list_uncertainity_avoidance_random)
      input_list_nearest_random = alphabetical_list_creation(nearest_random_list_uncertainity_avoidance)
      input_list_human = alphabetical_list_creation(list_uncertainity_avoidance)
      input_list_nearest_human = alphabetical_list_creation(nearest_human_list_uncertainity_avoidance)
        
    company_df = total_company_count_df.copy()  # This make a copy of data frame
    
    #Below lines will find the number of companies aligned to the respective left word in antonym pair
    company_df = company_count(company_df,input_list_random,polar_embedding)
    company_df = company_count(company_df,input_list_nearest_random,polar_embedding)
    company_df = company_count(company_df,input_list_human,polar_embedding)
    company_df = company_count(company_df,input_list_nearest_human,polar_embedding)

    #Below lines will find the total score based on the left word and final give a ranking
    company_df = polar_ranking(left_polar_list_random,"Total Score Random","Polar Rank R",company_df)
    company_df = polar_ranking(left_polar_list_nearest_random,"Total Score Nearest Random","Polar Rank Nearest R",company_df)
    company_df = polar_ranking(left_polar_list_human,"Total Score Human","Polar Rank H",company_df)
    company_df = polar_ranking(left_polar_list_nearest_human,"Total Score Nearest Human","Polar Rank Nearest H",company_df)

    length = len(left_polar_list_random) + len(left_polar_list_nearest_random) + len(left_polar_list_human) + len(left_polar_list_nearest_human)
    company_df.drop(company_df.iloc[:, 2:2 + (length) * 2], axis=1, inplace=True)


    hofstede_df = hofstede_df[hofstede_df.iloc[:, :] != "#NULL!"]
    hofstede_df.dropna(axis=0)

    # This merge the company dataframe and Hofstede dataframe over the common column Country
    

    merged_df = pd.merge(company_df, hofstede_df, how='left', on='Country')
    ranking_list = []
    for i in range(1, len(merged_df[dim_index]) + 1):
        ranking_list.append(i)
    merged_df = merged_df.sort_values(by=[dim_index], ascending=False)
    merged_df[dim_ranking] = ranking_list

    correlation = st.sidebar.checkbox('correlation')
    pshs = st.sidebar.checkbox('Polar score vs Hofstede score')

    if(correlation):
        # Below are the correlation plot 
        fig1 = plt.figure(figsize = (10,7))
        plt.subplot(2, 2, 1)
        sns.regplot(x=merged_df[dim_ranking], y=merged_df["Polar Rank R"])
        plt.subplot(2, 2, 2)
        sns.regplot(x=merged_df[dim_ranking], y=merged_df["Polar Rank Nearest R"])
        plt.subplot(2, 2, 3)
        sns.regplot(x=merged_df[dim_ranking], y=merged_df["Polar Rank H"])
        plt.subplot(2, 2, 4)
        sns.regplot(x=merged_df[dim_ranking], y=merged_df["Polar Rank Nearest H"])
        st.pyplot(fig1)
        
    if(pshs):
    # Below is the Hofstede dimension score and our score we got for each of the 4 list

        fig = go.Figure()
        fig = make_subplots(rows=2, cols=2)

        fig.add_trace(go.Bar(x=merged_df["Country"] , y=merged_df[dim_index].astype(int), name = dim_index),1,1)  
        fig.add_trace(go.Bar(x=merged_df["Country"] , y=merged_df["Total Score Random"].astype(int), name = "Random Polar Score"),1,1)  

        fig.add_trace(go.Bar(x=merged_df["Country"] , y=merged_df[dim_index].astype(int), name = dim_index),1,2)  
        fig.add_trace(go.Bar(x=merged_df["Country"] , y=merged_df["Total Score Nearest Random"].astype(int), name = "Nearest Random Polar Score"),1,2)  

        fig.add_trace(go.Bar(x=merged_df["Country"] , y=merged_df[dim_index].astype(int), name = dim_index),2,1)  
        fig.add_trace(go.Bar(x=merged_df["Country"] , y=merged_df["Total Score Human"].astype(int), name = "Human Polar Score"),2,1)  

        fig.add_trace(go.Bar(x=merged_df["Country"] , y=merged_df[dim_index].astype(int), name = dim_index),2,2)  
        fig.add_trace(go.Bar(x=merged_df["Country"] , y=merged_df["Total Score Nearest Human"].astype(int), name = "Nearest Human Polar Score"),2,2) 
        
        fig.update_layout(height=600, width=800, title_text="Polar score vs Hofstede score")
        st.plotly_chart(fig)

    MAE = mean_absolute_error_rank(merged_df,dim_ranking)
    MAE_of_Score = mean_absolute_error_rank(merged_df,dim_ranking)
    correlation = correlation_calc(merged_df,dim_ranking)


    # The below code creates a data frame with the results
    eval_data = {"Mean Absolute Error of Rank" : MAE,
                  "Correlation" : correlation,
                "Mean Absolute Error of Score" : MAE_of_Score
                }

    eval_df = pd.DataFrame(eval_data, index =["Random List", "Nearest Random List","Human Made List","Nearest to Human Made List"])
    eval_df.head()

    corr = merged_df.corr()
    st.write(corr.style.background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1).highlight_null(null_color='#f1f1f1').set_precision(2))

if(check == 'PCA'):
    pre_trained = st.sidebar.selectbox("Select a pre-trained model", ('Wikipedia','Google News', 'Reddit', 'Twitter' ))

    if(pre_trained == 'Wikipedia'):
        antonyms = st.sidebar.selectbox("Select a antonym set", ('Business Antonyms','Original Polar Antonyms' ))
        if(antonyms == 'Business Antonyms'):
            company_url = "https://docs.google.com/spreadsheets/d/1svpPZ2zw7G_iQsSTuogRAFgReO8X_CrHp7fL1uitLBo/edit?usp=sharing"
            common_words = st.sidebar.selectbox("Add common words", ('100 Common words','1000 common words' ))
            if(common_words == '100 Common words'):
                common = "https://docs.google.com/spreadsheets/d/1xUPWmpttnUx5Mf_k7y4zZxBTKKkXcEAN-lfydDPgoUM/edit?usp=sharing"
            elif(common_words  == '1000 common words'):
                common = "https://docs.google.com/spreadsheets/d/1XpFjXCUbkcqYDza1-cUtZvcli3NhdWGyiHMd6S8rUMw/edit?usp=sharing"
        if(antonyms == 'Original Polar Antonyms'):
            company_url = "https://docs.google.com/spreadsheets/d/1pgs3qW8k47xFVKN4TEjXlpfksTJaERYCNKaQFxekUcM/edit?usp=sharing"
            common_words = st.sidebar.selectbox("Add common words", ('100 Common words','1000 common words' ))
            if(common_words == '100 Common words'):
                common = "https://docs.google.com/spreadsheets/d/1T4LMmiC-QmDEBhF1uy8OmtlrBYQ23aRTykmhbG3Y5cE/edit?usp=sharing"
            elif(common_words  == '1000 common words'):
                common = "https://docs.google.com/spreadsheets/d/1vzvJqSRpnZtdjDOi1DAOCOvkBAsqokj83ZnSWBvbVLU/edit?usp=sharing"  

    elif(pre_trained == 'Google News'):
        antonyms = st.sidebar.selectbox("Select a antonym set", ('Business Antonyms','Original Polar Antonyms' ))
        if(antonyms == 'Business Antonyms'):
            company_url = "https://docs.google.com/spreadsheets/d/1yKz9rcSH31DnNG989S_klSDcAqcTxq4i1Jhk73wl4ug/edit?usp=sharing"
            common_words = st.sidebar.selectbox("Add common words", ('100 Common words','1000 common words' ))
            if(common_words == '100 Common words'):
                common = "https://docs.google.com/spreadsheets/d/1zp6Gd8vasd82brthouaNUPvd0ZV2Df5sIepwLXtORck/edit?usp=sharing"
            elif(common_words  == '1000 common words'):
                common = "https://docs.google.com/spreadsheets/d/1_FvoyeT7nO70jukT5HUe097w29NyQoCISIfpRDv5WRs/edit?usp=sharing"
        if(antonyms == 'Original Polar Antonyms'):
            company_url = "https://docs.google.com/spreadsheets/d/1ff_GKJnBDBb4XaMx64Y5TqYxWVqVW_SBlQsFRBqO9Yc/edit?usp=sharing"
            common_words = st.sidebar.selectbox("Add common words", ('100 Common words','1000 common words' ))
            if(common_words == '100 Common words'):
                common = "https://docs.google.com/spreadsheets/d/1tulznmXgifzNVubJ-UU3GK2D5pwREbqBYm9WIAnLQz8/edit?usp=sharing"
            elif(common_words  == '1000 common words'):
                common = "https://docs.google.com/spreadsheets/d/10PgRzLfvhpWg3XH7YjTcazXv_o1OJfBtP7bnIZ1dfEY/edit?usp=sharing"

    elif(pre_trained == 'Reddit'):
        antonyms = st.sidebar.selectbox("Select a antonym set", ('Business Antonyms','Original Polar Antonyms' ))
        if(antonyms == 'Business Antonyms'):
            company_url = "https://docs.google.com/spreadsheets/d/1-MvGo9EgEDTc9joPBLPtiAnMSTQHYm1MrETD5zp7Tfo/edit?usp=sharing"
            common_words = st.sidebar.selectbox("Add common words", ('100 Common words','1000 common words' ))
            if(common_words == '100 Common words'):
                common = "https://docs.google.com/spreadsheets/d/1ElHqYvfqbGXDEBLg1TUjyvW7ozSuzheY-x4fPq5xu5o/edit?usp=sharing"
            elif(common_words  == '1000 common words'):
                common = "https://docs.google.com/spreadsheets/d/1v51TTUPz8uAbWni9U9tWTkQwFGXfV1p8WWLBaXq6hyo/edit?usp=sharing"
        if(antonyms == 'Original Polar Antonyms'):
            company_url = "https://docs.google.com/spreadsheets/d/1H1pvjf5ppMsK8RGSPsXqct7-Q49GarWmZ6FayYaCln4/edit?usp=sharing"
            common_words = st.sidebar.selectbox("Add common words", ('100 Common words','1000 common words' ))
            if(common_words == '100 Common words'):
                common = "https://docs.google.com/spreadsheets/d/14PZK-GOppWLNuEXXdBI_JCtldZfpyYErSC7IDu532V8/edit?usp=sharing"
            elif(common_words  == '1000 common words'):
                common = "https://docs.google.com/spreadsheets/d/1GSU-SXjWgiaN9sBg6PJUMcLli8xCjoo28a81peqzG-Y/edit?usp=sharing" 

    elif(pre_trained == 'Twitter'):
        antonyms = st.sidebar.selectbox("Select a antonym set", ('Business Antonyms','Original Polar Antonyms' ))
        if(antonyms == 'Business Antonyms'):
            company_url = "https://docs.google.com/spreadsheets/d/13dh-3rnJ1HiU5V1Juoc2GDxXiFC6iWdaSWPsAieSVAM/edit?usp=sharing"
            common_words = st.sidebar.selectbox("Add common words", ('100 Common words','1000 common words' ))
            if(common_words == '100 Common words'):
                common = "https://docs.google.com/spreadsheets/d/1i8GsCcxV0tbe-T9GdmB-ML9lpZG5i9BsLKVdozkhzjc/edit?usp=sharing"
            elif(common_words  == '1000 common words'):
                common = "https://docs.google.com/spreadsheets/d/1t9McYMoUFB4BVsri73rBAuzx7xPqSNWxpWkTpuEPKyE/edit?usp=sharing"
        if(antonyms == 'Original Polar Antonyms'):
            company_url = "https://docs.google.com/spreadsheets/d/1b-YHAzrDp63g2PQ3sLMGbYER3_ir0rPgluz6Sww3b-k/edit?usp=sharing"
            common_words = st.sidebar.selectbox("Add common words", ('100 Common words','1000 common words' ))
            if(common_words == '100 Common words'):
                common = "https://docs.google.com/spreadsheets/d/1St8SPiAPq4InWKY3Yp1tXSNghPraY_fnSzQEU5Nomkc/edit?usp=sharing"
            elif(common_words  == '1000 common words'):
                common = "https://docs.google.com/spreadsheets/d/1LJ_CYDiMifY397X-FtSnTYvbBqyg3CZniOLj2dvAjEg/edit?usp=sharing"        

    conn = connect()

    common = conn.execute(f'SELECT * FROM "{common}"')
    company_url = conn.execute(f'SELECT * FROM "{company_url}"')
    company_url = pd.DataFrame(company_url)
    common = pd.DataFrame(common)

    new_df=pd.concat([common, company_url])  
    df_cluster=new_df.loc[:,new_df.columns!='name']          
    dimension = st.sidebar.selectbox("Select Dimensions", ('2D','3D' ))

    if(dimension == '2D'):
        eps = st.sidebar.slider('Select epsilon', 0.0, 1.0, 0.4)
        min_samples = st.sidebar.slider('Select minimum samples', 1, 10, 5)
        dbscan = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples)#high eps low samples only clusters common, 
        cluster_labels = dbscan.fit_predict(df_cluster)

        two_dim = PCA(random_state=0).fit_transform(df_cluster)[:,:2]
        df_cluster[['two_dim1','two_dim2']]=two_dim.tolist()
        df_cluster['cluster']=cluster_labels
        df_cluster['Unnamed: 0']=new_df['name']

        display_pca_scatterplot_2D(df_cluster[:])

    elif(dimension == '3D'):

        eps = st.sidebar.slider('Select epsilon', 0.0, 1.0, 0.4)
        min_samples = st.sidebar.slider('Select minimum samples', 1, 10, 5)
        dbscan = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples)#high eps low samples only clusters common, 
        cluster_labels = dbscan.fit_predict(df_cluster)
        
        three_dim = PCA(random_state=0).fit_transform(df_cluster)[:,:3]
        df_cluster[['three_dim1','three_dim2','three_dim3']]=three_dim.tolist()
        df_cluster['cluster']=cluster_labels
        df_cluster['Unnamed: 0']=new_df['name']

        display_pca_scatterplot_3D(df_cluster[:])

