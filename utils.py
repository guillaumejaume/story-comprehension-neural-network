#import tensorflow as tf
import numpy as np
import csv
import re
import codecs
import random
from random import randint

import scipy

from story import Story


def load_story_ids(filename):
    """ Load a file and read it line-by-line
    Parameters:
    -----------
    filename: string
    path to the file to load

    Returns:
    --------
    raw: list of sentences
    """
    f = open(filename, "r")
    story_ids = [line[:-1] for line in f]
    f.close()
    return story_ids


def load_bin(filename, dim_embeddings):
    """ Read a bin file and store into list
    Parameters:
    -----------
    - filename: string
        path to the file to load

    - embeddings_dim: int

    Returns:
    --------
    data: list of floats
    """
    f = open(filename, 'rb')
    temp = np.fromfile(f, 'f')

    number_of_embeddings = int(len(temp)/dim_embeddings)

    data = []
    for i in range(number_of_embeddings):
        d = temp[i * dim_embeddings:(i+1) * dim_embeddings]
        data.append(d)
    return data


def load_and_process_text_data(filename, for_testing=False, is_labeled = True):
    """ Load a file and create a list of sentences
    Parameters:
    -----------
    filename: string
    path to the file to load

    Returns:
    --------
    raw: list of sentences
    """
    list_of_sentences = []
    first_line = True

    with open(filename, encoding = 'ISO-8859-1') as f:
        reader = csv.reader(f)
        story_id = 0
        for row in reader:
            if first_line and row[0] == "InputStoryId":
                first_line = False
            else:
                if not for_testing:
                    id = row[0]
                    s1 = row[2]
                    s2 = row[3]
                    s3 = row[4]
                    s4 = row[5]
                    re = row[6]
                    list_of_sentences.append(Story(id, s1, s2, s3, s4, re))

                else:
                    if is_labeled:
                        if int(row[7]) == 1:
                            id = row[0]
                            s1 = row[1]
                            s2 = row[2]
                            s3 = row[3]
                            s4 = row[4]
                            re = row[5]
                            we = row[6]
                        else:
                            id = row[0]
                            s1 = row[1]
                            s2 = row[2]
                            s3 = row[3]
                            s4 = row[4]
                            re = row[6]
                            we = row[5]
                    else:
                        id = str(story_id)
                        s1 = row[0]
                        s2 = row[1]
                        s3 = row[2]
                        s4 = row[3]
                        re = row[4]
                        we = row[5]
                        story_id = story_id + 1

                    list_of_sentences.append(Story(id, s1, s2, s3, s4, re, we))

    return list_of_sentences


def load_embeddings(embeddings_input_path, embeddings_dim):
    """ Load the embeddings from file
    Parameters:
    -----------
    - embeddings_input_file_path: string
        path to the directory to load the embeddings from

    embeddings_dim: int
    Returns:
    --------
    embeddings: dictionary of embeddings
    """
    embeddings = {}
    story_ids = load_story_ids(embeddings_input_path + 'id.txt')
    for i in range(len(story_ids)):
        emb = load_bin(embeddings_input_path + story_ids[i], embeddings_dim)
        embeddings[story_ids[i]] = emb
    return embeddings


def select_embeddings_for_model(embeddings, model_type, has_right_ending=True, embedding_dimension=4800):
    """ Select embeddings for a specific sentence/group of sentences based on the type of analysis
    Parameters:
    -----------
    - embeddings: dictionary
        dictionary of embeddings {key: [emb_s1, emb_s2, emb_s3, emb_s4, emb_end]}

    - model_type: string
        decides the embeddings that will be selected # (full - all 5 sentences), (plot = first 4 sentences), (last_sentence = the 4th sentence)

    - has_right_ending: bool
        decides if has the right ending or not. If not, "wrong" is appended to each key in the dictionary

    - embedding_dimension: int
        the dimension of the embedding

    Returns:
    --------
    embeddings: dictionary of embeddings
    """
    embeddings_slice = {}
    for key, value in embeddings.items():
        if not has_right_ending:
            key = key + "wrong"
        if model_type == "full":
            embeddings_slice[key] = value
        elif model_type == "plot":
            embeddings_slice[key] = value[:4]
        elif model_type == "last_sentence":
            embeddings_slice[key] = value[3]
        elif model_type == "no_context":
            embeddings_slice[key] = [0] * embedding_dimension

    return embeddings_slice


def select_endings(embeddings, has_right_ending = True, modify_key = True):
    """ Select ending embeddings for a specific sentence/group of sentences
    Parameters:
    -----------
    embeddings: dictionary
    dictionary of embeddings {key: [emb_s1, emb_s2, emb_s3, emb_s4, emb_end (, emb_end2)]}

    has_right_ending: bool
    control whether the function is using the right ending or the wrong ending

    modify_key: bool
    used to modify the key if wrong ending

    Returns:
    --------
    ending_embeddings: dictionary of embeddings

    labels: dictionary
    dictionary of labels
    """
    ending_embeddings = {}
    labels = {}

    for key, value in embeddings.items():
        if has_right_ending:
            ending_embeddings[key] = value[4]
            labels[key] = 1
        else:
            if modify_key:
                key = key + "wrong"
            ending_embeddings[key] = value[5]
            labels[key] = 0

    return ending_embeddings, labels


def select_random_ending(story_embeddings):
    """ Select a random ending for each sentence
    Parameters:
    -----------
    story_embeddings: dictionary
    dictionary of embeddings {key: [emb_s1, emb_s2, emb_s3, emb_s4, emb_end]}

    Returns:
    --------
    random_ending_embeddings: dictionary
    dictionary with random endings

    labels: dictionary
    dictionary of labels
    """
    random_ending_embeddings = {}
    labels = {}
    for i, key in enumerate(story_embeddings):
        random_key = random.choice(list(story_embeddings.keys()))
        while random_key == key:
            random_key = random.choice(list(story_embeddings.keys()))
        random_ending_embeddings[key + "wrong"] = story_embeddings[key][randint(0, 3)]
        labels[key + "wrong"] = 0
    return random_ending_embeddings, labels


def convert_training_dictionaries_to_lists(beginning_of_story_embeddings, ending_embeddings, labels):
    """ Convert dictionaries to list
    Parameters:
    -----------
    beginning_of_story_embeddings: dictionary
    dictionary of embeddings {key: value}

    ending_embeddings: dictionary
    dictionary of embeddings {key: value}

    labels: dictionary
    dictionary of labels {key: value}

    Returns:
    --------
    beginning_of_story_embeddings_list: list
    list of embeddings

    ending_embeddings_list: list
    list of embeddings

    labels_list: list
    list of embeddings

    """
    beginning_of_story_embeddings_list = []
    ending_embeddings_list = []
    labels_list = []
    for i, key in enumerate(beginning_of_story_embeddings):
        beginning_of_story_embeddings_list.append(beginning_of_story_embeddings[key])
        ending_embeddings_list.append(ending_embeddings[key])
        labels_list.append(labels[key])

    return beginning_of_story_embeddings_list,  ending_embeddings_list, labels_list


def convert_test_dictionaries_to_lists(beginning_of_story_embeddings, right_ending_embeddings, wrong_ending_embeddings, right_labels, wrong_labels):
    """ Convert dictionaries to list
     Parameters:
     -----------
     beginning_of_story_embeddings: dictionary
     dictionary of embeddings {key: value}

     right_ending_embeddings: dictionary
     dictionary of embeddings {key: value}

     wrong_ending_embeddings: dictionary
     dictionary of embeddings {key: value}

     right_labels: dictionary
     dictionary of labels {key: value}

     wrong_labels: dictionary
     dictionary of labels {key: value}

     Returns:
     --------
     beginning_of_story_embeddings_list: list
     list of embeddings

     right_ending_embeddings_list: list
     list of embeddings

     wrong_ending_embeddings_list: list
     list of embeddings

     right_labels_list: list
     list of labels

     wrong_labels_list: list
     list of labels
    """
    beginning_of_story_embeddings_list = []
    right_ending_embeddings_list = []
    wrong_ending_embeddings_list = []
    right_labels_list = []
    wrong_labels_list = []

    for i, key in enumerate(beginning_of_story_embeddings):
        beginning_of_story_embeddings_list.append(beginning_of_story_embeddings[key])
        right_ending_embeddings_list.append(right_ending_embeddings[key])
        wrong_ending_embeddings_list.append(wrong_ending_embeddings[key])
        right_labels_list.append(right_labels[key])
        wrong_labels_list.append(wrong_labels[key])

    return beginning_of_story_embeddings_list, right_ending_embeddings_list, wrong_ending_embeddings_list, right_labels_list, wrong_labels_list

def generate_data(all_embeddings, neg_samples_file=''):

    stories = []
    true_endings = []
    wrong_endings = []

    if neg_samples_file:
        closest_ending_pairs = get_closest_pairs(neg_samples_file)

    for key, val in all_embeddings.items():
        # @TODO Reorganize later, brute force now
        if neg_samples_file and key in closest_ending_pairs:
            story = val[:4]  # list of 4 array of 4800 float
            true_ending = val[4]  # true ending
            wrong_ending = all_embeddings[closest_ending_pairs[key]][4]
            stories.append(story)
            true_endings.append(true_ending)
            wrong_endings.append(wrong_ending)
        elif not neg_samples_file:
            story = val[:4]  # list of 4 array of 4800 float
            true_ending = val[4]  # true ending
            if len(val) > 5:
                wrong_ending = val[5]
                wrong_endings.append(wrong_ending)
            else:
                wrong_ending = all_embeddings[random.choice(all_keys)][4]
                wrong_endings.append(wrong_ending)
            stories.append(story)
            true_endings.append(true_ending)

    return np.asarray(stories), np.asarray(true_endings), np.asarray(wrong_endings)

def get_closest_pairs(neg_samples_file):
    all_pairs = {}
    with open(neg_samples_file) as f:
        for line in f:
            line = line.rstrip()
            keys = line.split(' ')
            all_pairs[keys[0]] = keys[1]
    return all_pairs


def find_closest_ending(current_key, all_keys, all_embeddings):
    current_emb = all_embeddings[current_key]

    min_dist = 1
    min_key = ''
    for key in all_keys:
        dist = scipy.spatial.distance.cosine(current_emb[4], all_embeddings[key][4])
        if dist < min_dist:
            min_dist = dist
            min_key = key
    return min_key

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def write_list_to_file(list_to_write, file_name):
    file = open(file_name, "w")
    for item in list_to_write:
        file.write("%s\n" % item)

    file.close()


def split_data_for_validation(input_file, validation_ids_output_file, test_ids_output_file, val_sample_percentage):
    """ Split the data for validation
    Parameters:
    -----------
    input_file: string
    csv file containing the data

    validation_ids_output_file: string
    file to write the ids that will be used for validating the model

    test_ids_output_file: string
    file to write the ids that will be used for testing the model

    val_sample_percentage: float
    percentage used to split the data
    """
    ids = []
    first_line = True
    with open(input_file) as f:
        reader = csv.reader(f)
        for row in reader:
            if first_line:
                first_line = False
            else:
                ids.append(row[0])

    val_sample_index = -1 * int(val_sample_percentage * float(len(ids)))
    ids_list = list(ids)
    random.shuffle(ids_list)

    test_ids = ids_list[:val_sample_index]
    validation_ids = ids_list[val_sample_index:]

    write_list_to_file(validation_ids, validation_ids_output_file)
    write_list_to_file(test_ids, test_ids_output_file)


def shuffle_data(a, b, c):
    """ Shuffle data
    Parameters:
    -----------
    a: list

    b: list

    c: list

    Returns:
    Shuffled data
    """
    data = list(zip(a, b, c))
    random.shuffle(data)
    a, b, c = zip(*data)

    return a, b, c















