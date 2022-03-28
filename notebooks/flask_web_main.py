from flask import Flask
from flask import request

import gensim
from numpy import linalg
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm_notebook as tqdm
import time
from random import shuffle
import sys
import nltk
from nltk.corpus import wordnet
import gc
from collections import defaultdict
import random
import json
import os
import pandas as pd
import random
import scipy
import torch
import subprocess
from flask import request, jsonify, json  # import jsonify

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

from gensim.test.utils import datapath

app = Flask(__name__)


@app.route("/")
def index():
    company = request.args.get("company", "")
    if company:
        embed = make_polar_dict(company, antonym_500, ticker_new_embedding, True)
    else:
        embed = ""
    return (
            """<form action="" method="get">
                Company ticker: <input type="text" name="company">
                <input type="submit" value="Get Embeddings">
            </form>"""
            + "POLAR Embeddings: "
            + str(embed)
    )


model_gn = gensim.models.KeyedVectors.load_word2vec_format(
    '/Users/stjepankusenic/POLAR_WEBE/data/raw/glove_norm_300.mod', binary=True)  # ss

current_model = model_gn

list_antonym = []

with open('/Users/stjepankusenic/POLAR_WEBE/data/raw/Antonym_sets/LenciBenotto.val') as fp:
    for line in fp:
        parts = line.split()
        if parts[3] == 'antonym':
            word1 = parts[0].split('-')[0]
            word2 = parts[1].split('-')[0]
            if word1 in current_model and word2 in current_model:
                list_antonym.append((word1.strip().lower(), word2.strip().lower()))

with open('/Users/stjepankusenic/POLAR_WEBE/data/raw/Antonym_sets/LenciBenotto.test') as fp:
    for line in fp:
        parts = line.split()
        if parts[3] == 'antonym':
            word1 = parts[0].split('-')[0]
            word2 = parts[1].split('-')[0]
            if word1 in current_model and word2 in current_model:
                list_antonym.append((word1.strip().lower(), word2.strip().lower()))

with open('/Users/stjepankusenic/POLAR_WEBE/data/raw/Antonym_sets/EVALution.val') as fp:
    for line in fp:
        parts = line.split()
        if parts[3] == 'antonym':
            word1 = parts[0].split('-')[0]
            word2 = parts[1].split('-')[0]
            if word1 in current_model and word2 in current_model:
                list_antonym.append((word1.strip().lower(), word2.strip().lower()))

with open('/Users/stjepankusenic/POLAR_WEBE/data/raw/Antonym_sets/EVALution.test') as fp:
    for line in fp:
        parts = line.split()
        if parts[3] == 'antonym':
            word1 = parts[0].split('-')[0]
            word2 = parts[1].split('-')[0]
            if word1 in current_model and word2 in current_model:
                list_antonym.append((word1.strip().lower(), word2.strip().lower()))

list_antonym = list(dict.fromkeys(list_antonym).keys())

similarity_matrix = defaultdict(list)
for each_pair in list_antonym:
    word1 = each_pair[0]
    word2 = each_pair[1]
    if word1 < word2:
        similarity_matrix[word1].append(word2)
    else:
        similarity_matrix[word2].append(word1)

all_similarity = defaultdict(dict)
for each_key in similarity_matrix:
    for each_value in similarity_matrix[each_key]:
        #         cosine_similarity([current_model[each_key]]
        all_similarity[each_key][each_value] = abs(
            cosine_similarity([current_model[each_key]], [current_model[each_value]])[0][0])

final_antonym_list = []
for index_counter, each_key in enumerate(all_similarity):
    #     print(each_key,all_similarity[each_key])
    listofTuples = sorted(all_similarity[each_key].items(), key=lambda x: x[1])
    #     print(listofTuples)
    final_antonym_list.append((each_key, listofTuples[0][0]))
print(len(final_antonym_list))

list_antonym = final_antonym_list

list_antonym = pd.read_pickle(r'/Users/stjepankusenic/POLAR_WEBE/data/interim/final_antonym_list')

num_antonym = 1468

## Find the antonym difference vectors
antonymy_vector = []
for each_word_pair in list_antonym:
    antonymy_vector.append(current_model[each_word_pair[0]] - current_model[each_word_pair[1]])
antonymy_vector = np.array(antonymy_vector)
print(antonymy_vector.shape)

random.seed(42)

t1 = np.array(antonymy_vector)
dimension_similarity_matrix = scipy.spatial.distance.cdist(np.array(antonymy_vector), np.array(antonymy_vector),
                                                           'cosine')
dimension_similarity_matrix = abs(1 - dimension_similarity_matrix)


def get_set_score(final_list, each_dim):
    final_output = 0.0
    for each_vec in final_list:
        final_output += dimension_similarity_matrix[each_vec][each_dim]
    return final_output / (len(final_list))


def select_subset_dimension(dim_vector, num_dim):
    working_list = np.array(dim_vector)

    working_position_index = [i for i in range(working_list.shape[0])]
    final_position_index = []

    print('working list is ready, shape', working_list.shape)
    sel_dim = random.randrange(0, working_list.shape[0])

    final_position_index.append(sel_dim)

    working_position_index.remove(sel_dim)

    for test_count in range(num_dim - 1):
        min_dim = None
        min_score = 1000
        for temp_index, each_dim in enumerate(working_position_index):
            temp_score = get_set_score(final_position_index, each_dim)
            if temp_score < min_score:
                min_score = temp_score
                min_dim = each_dim
        final_position_index.append(min_dim)
        working_position_index.remove(min_dim)
    return final_position_index


random_antonym_vector = [i for i in range(len(antonymy_vector))]
random.shuffle(random_antonym_vector)
print(len(random_antonym_vector))

orthogonal_antonymy_vector = np.array(select_subset_dimension(antonymy_vector, num_antonym))
print(orthogonal_antonymy_vector.shape)

embedding_size = antonymy_vector.shape[0]
print('The embedding size is', embedding_size)

variance_antonymy_vector_inverse = np.linalg.pinv(np.transpose(antonymy_vector))
variance_antonymy_vector_inverse = torch.tensor(variance_antonymy_vector_inverse)

embedding_matrix = []

current_model_tensor = torch.t(torch.tensor(current_model.wv.vectors))

var_list = [None for x in range(20)]  # variance for each antonym in each batch

for i in range(19):  # the first 19 batches, each of size 100k
    temp = torch.matmul(variance_antonymy_vector_inverse, current_model_tensor[:, 100000 * i:100000 * i + 100000])
    temp_var_mean = torch.var(temp, axis=1)
    var_list[i] = temp_var_mean.numpy()
    del temp

temp = torch.matmul(variance_antonymy_vector_inverse, current_model_tensor[:, 1900000:])
temp_var_mean = torch.var(temp, axis=1)
var_list[19] = temp_var_mean.numpy()
del temp

# using lazy approach. assume each batch is independent and the overall variance is the average variance over all batches

variance_list = np.mean(np.array(var_list), axis=0)

variance_antonymy_vector = [each for each in
                            sorted(range(len(variance_list)), key=lambda i: variance_list[i], reverse=True)]


def transform_to_antonym_space(current_model, output_file_path, binary, current_antonymy_vector_inverse):
    temp_dict = dict()

    embedding_size = current_antonymy_vector_inverse.shape[0]  ##CHANGE THIS ACCORDINGLY!!!
    print('New model size is', len(current_model), embedding_size)

    temp_file = None

    if binary:
        temp_file = open(output_file_path, 'wb')
        temp_file.write(str.encode(str(len(current_model)) + ' ' + str(embedding_size) + '\n'))
    else:
        temp_file = open(output_file_path, 'w')
        temp_file.write(str(len(current_model)) + ' ' + str(embedding_size) + '\n')

    total_words = 0
    for each_word in current_model:
        total_words += 1
        if binary:
            temp_file.write(str.encode(each_word + ' '))
        else:
            temp_file.write(each_word + ' ')

        new_vector = np.matmul(current_antonymy_vector_inverse, current_model[each_word])

        new_vector = new_vector / linalg.norm(new_vector)
        temp_dict[each_word] = new_vector

        if binary:
            temp_file.write(new_vector)
            temp_file.write(str.encode('\n'))
        else:
            temp_file.write(str(new_vector))
            temp_file.write('\n')

    temp_file.close()
    return temp_dict


def standard_normal_dist_model(model, new_filename):
    embedding_matrix = []
    embedding_vocab = []

    temp_file = open(new_filename, 'wb')
    temp_file.write(str.encode(str(model.vectors.shape[0]) + ' ' + str(model.vectors.shape[1]) + '\n'))

    for each_word in model.vocab:
        embedding_matrix.append(model[each_word])
        embedding_vocab.append(each_word)

    embedding_matrix = np.array(embedding_matrix)

    print('The shape of embedding matrix is {}'.format(embedding_matrix.shape))

    norm_embedding_matrix = (embedding_matrix - embedding_matrix.mean(0)) / embedding_matrix.std(0)

    for word_counter, each_word in enumerate(embedding_vocab):
        #         assert each_word==embedding_vocab[word_counter],'Not matching!!!'

        temp_file.write(str.encode(each_word + ' '))
        new_vector = norm_embedding_matrix[word_counter]
        temp_file.write(new_vector)
        temp_file.write(str.encode('\n'))

    del embedding_matrix
    del embedding_vocab
    temp_file.close()


def generate_embedding_path(current_model, embedding_path, binary, antonym_vector, curr_dim):
    curr_antonym_vector = antonymy_vector[antonym_vector[:curr_dim]]
    curr_antonymy_vector_inverse = np.linalg.pinv(np.transpose(curr_antonym_vector))
    new_embedding_dict = transform_to_antonym_space(current_model, embedding_path, binary, curr_antonymy_vector_inverse)

    return new_embedding_dict


dim_size = 500  # Number of POLAR dimenions
antonym_vector_method = variance_antonymy_vector  # orthogonal_antonymy_vector, variance_antonymy_vector, random_antonym_vector

company = pd.read_csv('/Users/stjepankusenic/POLAR_WEBE/data/interim/glove-ticker-fortune1000-us.csv')
ticker_list = company['Ticker'].str.lower()

ticker_word_embedding = dict()
for ticker in ticker_list:
    ticker_word_embedding[ticker] = current_model[ticker]

ticker_new_embedding = generate_embedding_path(ticker_word_embedding, 'test_run', True, antonym_vector_method, 500)

antonym_500 = [list_antonym[x] for x in antonym_vector_method[:500]]


def make_polar_dict(company_name, antonym, embedding, top_n=False, n=10):
    temp_dict = dict()
    temp_polar = embedding[company_name]
    temp_out = dict()

    if top_n:
        idx = np.argsort([abs(x) for x in temp_polar])[-n:]
        for i in idx:
            print(antonym[i], temp_polar[i], '\n')
            temp_out[antonym[i]] = temp_polar[i]
        return temp_out

    if len(antonym) == len(temp_polar):
        for a in range(len(antonym)):
            temp_dict[antonym[a]] = temp_polar[a]
        return temp_dict


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
