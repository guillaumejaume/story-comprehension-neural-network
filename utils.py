#import tensorflow as tf
import numpy as np
import csv
import pprint
import re
import skipthoughts
import codecs

import sys
reload(sys)
sys.setdefaultencoding('UTF8')

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

def load_and_process_data(filename):
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
                list_of_sentences.append(Story(row[0], #storyid
                                              row[1], #storytitle
                                              row[2], #sentence1
                                              row[3], #sentence2
                                              row[4], #sentence3
                                              row[5], #sentence4
                                              row[6]  #sentence5
                                              ))

    return list_of_sentences

def load_embeddings(embeddings_input_path, embeddings_id_input_file):
    """ Load the embeddings from file
       Parameters:
       -----------
       embeddings_input_file_path: string
       path to the directory to load the embeddings from

       embeddings_id_input_file: string
       path to the file to load the embeddings id from

       type: string
       decides the embeddings that will be loaded # (full - all 5 sentences), (plot = first 4 sentences),  (last_sentence = the 4th sentence), (ending = the 5th sentence)
       Returns:
       --------
       embeddings: dictionary of embeddings

       embeddings_id: list
       list with the id of embeddings
    """
    embeddings = {}
    embeddings_id = load_raw_data(embeddings_id_input_file)
    for i in range(len(embeddings_id)):
        fin_emb = open(embeddings_input_path + embeddings_id[i], "r")
        emb = np.load(fin_emb)
        fin_emb.close()
        embeddings[embeddings_id[i]] = emb
    return embeddings, embeddings_id

def select_embeddings(embeddings, type):
    """ Select embeddings for a specific sentence/group of sentences based on the type of analysis
       Parameters:
       -----------
       embeddings: dictionary
       dictionary of embeddings {key: [emb_s1, emb_s2, emb_s3, emb_s4, emb_end]}

       type: string
       decides the embeddings that will be selected # (full - all 5 sentences), (plot = first 4 sentences), (last_sentence = the 4th sentence), (ending = the 5th sentence)
       Returns:
       --------
       embeddings: dictionary of embeddings
    """
    embeddings_slice = {}
    for key, value in embeddings.items():
        if type == "full":
            embeddings_slice[key] = value
        elif type == "plot":
            embeddings_slice[key] = value[:4]
        elif type == "last_sentence":
            embeddings_slice[key] = value[3]
        elif type == "ending":
            embeddings_slice[key] = value[4]

    return embeddings_slice





