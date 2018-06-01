from math import sqrt
from joblib import Parallel, delayed
import utils
import os
import numpy as np
from itertools import islice
import time

def generate_neighbours(embeddings, file, threadId):
    f = open(file + str(threadId) + ".txt", "a")
    start_time = time.time()
    i = 0
    for key, val in embeddings.items():
        keys = list(embeddings.keys())
        keys.remove(key)
        closest_key = utils.find_closest_ending(key, keys, embeddings)
        wrong_ending = embeddings[closest_key][4]
        #print(str(threadId) + ' current key: ', key)
        #print(' closest key: ', closest_key)
        #print('\n')
        neighbours = key + " " + closest_key
        f.write("%s\n" % neighbours)
        i = i + 1
        if i % 1000 == 0:
            print(str(threadId ) + " " + str(i) + " iterations")
    f.write("--- %s seconds ---" % str(time.time() - start_time))
    f.write("--- %s seconds for ---" % str(time.time() - start_time))
    f.close()
    print("--- %s seconds for ---" % str(time.time() - start_time))


def divide_data(data, size):
    divided_data = []
    it = iter(data)

    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}


training_embeddings_dir = "./data/embeddings_training/"
validation_embeddings_dir = "./data/embeddings_validation/"
embedding_dim = 4800
num_threads = 4
training_output_file = "training_neighbours_"
validation_output_file = "validation_neighbours_"
# load training embeddings
all_training_embeddings = utils.load_embeddings(training_embeddings_dir, embedding_dim)
# load validation embeddings
all_validation_embeddings = utils.load_embeddings(validation_embeddings_dir, embedding_dim)

samples_per_thread = int((len(all_training_embeddings) - 1)/num_threads) + 1
divided_training_embeddings = list(divide_data(all_training_embeddings, samples_per_thread))
all_training_embeddings = []
samples_per_thread = int((len(all_validation_embeddings) - 1)/num_threads) + 1
divided_validation_embeddings = list(divide_data(all_validation_embeddings, samples_per_thread))
all_validation_embeddings = []

Parallel(n_jobs=num_threads)(generate_neighbours(divided_training_embeddings[i], training_output_file, i) for i in range(num_threads))
Parallel(n_jobs=num_threads)(generate_neighbours(divided_validation_embeddings[i], validation_output_file, i) for i in range(num_threads))
