#import tensorflow as tf
import numpy as np
import csv
import re
import codecs
import random
from random import randint

from story import Story

def load_raw_data(filename):
    """ Load a file and read it line-by-line
    Parameters:
    -----------
    filename: string
    path to the file to load

    Returns:
    --------
    raw: list of sentences
    """
    file = open(filename, "r")
    raw_data = [line[:-1] for line in file]
    file.close()
    return raw_data

def load_numerical_data_in_list(filename, num_embeddings, dim_embeddings):
    """ Load a float list from a binary file
        Parameters:
        -----------
        filename: string
        path to the file to load

        num_embeddings_per_story: int
        number of embeddings per story

        embeddings_dim: int
        the dimension of the embedding

        Returns:
        --------
        data: list of floats
        """
    f = open(filename, 'rb')
    temp = np.fromfile(f, 'f')
    data = []
    for i in range(num_embeddings):
        d = temp[i * dim_embeddings:(i+1) * dim_embeddings]
        data.append(d)
    return data

def load_and_process_text_data(filename, for_testing = False):
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
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            if first_line:
                first_line = False
            else:
                if not for_testing:
                    list_of_sentences.append(
                        Story(
                            row[0],
                            row[2], #sentence1
                            row[3], #sentence2
                            row[4], #sentence3
                            row[5], #sentence4
                            row[6]  #right_ending
                        )
                    )

                else:
                    if row[7] == 1:
                        list_of_sentences.append(
                            Story(
                                row[0],  # storyid
                                row[1],  # sentence1
                                row[2],  # sentence2
                                row[3],  # sentence3
                                row[4],  # sentence4
                                row[5],  # right_ending
                                row[6]   # wrong_ending
                            )
                        )
                    else:
                        list_of_sentences.append(
                            Story(
                                row[0],  # storyid
                                row[1],  # sentence1
                                row[2],  # sentence2
                                row[3],  # sentence3
                                row[4],  # sentence4
                                row[6],  # right_ending
                                row[5]   # wrong_ending
                            )
                        )

    return list_of_sentences

def load_embeddings(embeddings_input_path, embeddings_id_input_file, num_embeddings_per_story, embeddings_dim):
    """ Load the embeddings from file
       Parameters:
       -----------
       embeddings_input_file_path: string
       path to the directory to load the embeddings from

       embeddings_id_input_file: string
       path to the file to load the embeddings id from

       num_embeddings_per_story: int
       number of embeddings per story

       embeddings_dim: int
       the dimension of the embedding
       Returns:
       --------
       embeddings: dictionary of embeddings

       embeddings_id: list
       list with the id of embeddings
    """
    embeddings = {}
    embeddings_id = load_raw_data(embeddings_id_input_file)
    for i in range(len(embeddings_id)):
        emb = load_numerical_data_in_list(embeddings_input_path + embeddings_id[i], num_embeddings_per_story, embeddings_dim)
        embeddings[embeddings_id[i]] = emb

    return embeddings, embeddings_id

def select_embeddings_for_model(embeddings, model_type, has_right_ending = True, embedding_dimension = 4800):
    """ Select embeddings for a specific sentence/group of sentences based on the type of analysis
       Parameters:
       -----------
       embeddings: dictionary
       dictionary of embeddings {key: [emb_s1, emb_s2, emb_s3, emb_s4, emb_end]}

       model_type: string
       decides the embeddings that will be selected # (full - all 5 sentences), (plot = first 4 sentences), (last_sentence = the 4th sentence)

       has_right_ending: bool
       decides if has the right ending or not. If not, "wrong" is appended to each key in the dictionary

       embedding_dimension: int
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


def select_endings(embeddings, is_right_ending = True, modify_key = True):
    """ Select ending embeddings for a specific sentence/group of sentences
       Parameters:
       -----------
       embeddings: dictionary
       dictionary of embeddings {key: [emb_s1, emb_s2, emb_s3, emb_s4, emb_end]}

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
        if is_right_ending:
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


def generate_training_data(story_embeddings, story_type, generate_radom_ending = True):
    """ Generate data: embeddings for the beginning of sentence, for the end of sentence and the associated labels [1 for right/ 0 for wrong]
           Parameters:
           -----------
           story_embeddings: dictionary
           dictionary of embeddings {key: [emb_s1, emb_s2, emb_s3, emb_s4, emb_end1 (, emb_end2)]}

           story_type: string
           (full - all 5 sentences), (plot = first 4 sentences), (last_sentence = the 4th sentence)

           generate_radom_ending: bool
           if set, it is used to generate a random ending. Useful for datasets where's no wrong ending. The dataset is
           Returns:
           --------
           random_ending_embeddings: dictionary
           dictionary with random endings

           labels: dictionary
           dictionary of labels

    """
    beginning_of_story_embeddings = select_embeddings_for_model(story_embeddings, story_type)  # positive sampling
    beginning_of_story_embeddings.update(select_embeddings_for_model(story_embeddings, story_type, False)) # negative sampling
    ending_embeddings, labels = select_endings(story_embeddings) # positive sampling
    if generate_radom_ending:
        temp_ending_embeddings, temp_labels = select_random_ending(story_embeddings)
    else:
        is_right_ending = False # negative sampling
        temp_ending_embeddings, temp_labels = select_ending(story_embeddings, is_right_ending)

    ending_embeddings.update(temp_ending_embeddings)
    labels.update(temp_labels)
    beginning_of_story_embeddings, ending_embeddings, labels = convert_training_dictionaries_to_lists(beginning_of_story_embeddings, ending_embeddings, labels)

    return beginning_of_story_embeddings,  ending_embeddings, labels


def generate_test_data(story_embeddings, story_type):
    """ Generate data: embeddings for the beginning of sentence, for the end of sentence and the associated labels [1 for right/ 0 for wrong]
           Parameters:
           -----------
           story_embeddings: dictionary
           dictionary of embeddings {key: [emb_s1, emb_s2, emb_s3, emb_s4, emb_end1 (, emb_end2)]}

           story_type: string
           (full - all 5 sentences), (plot = first 4 sentences), (last_sentence = the 4th sentence)

           Returns:
           --------
           beginning_of_story_embeddings: list
           list of embeddings

           right_ending_embeddings: list
           list of embeddings

           wrong_ending_embeddings: list
           list of embeddings

           right_labels: list
           list of labels

           wrong_labels: list
           list of labels


    """
    is_right_ending = False # negative sampling
    modify_key = False

    beginning_of_story_embeddings = select_embeddings_for_model(story_embeddings, story_type)
    right_ending_embeddings, right_labels = select_endings(story_embeddings) # positive sampling
    wrong_ending_embeddings, wrong_labels = select_ending(story_embeddings, is_right_ending, modify_key)

    beginning_of_story_embeddings, right_ending_embeddings, wrong_ending_embeddings, right_labels, wrong_labels  = convert_test_dictionaries_to_lists(
        beginning_of_story_embeddings,
        right_ending_embeddings,
        wrong_ending_embeddings,
        right_labels,
        wrong_labels
    )

    return beginning_of_story_embeddings, right_ending_embeddings, wrong_ending_embeddings, right_labels, wrong_labels


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











