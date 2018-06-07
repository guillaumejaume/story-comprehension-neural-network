import numpy as np
import csv
import random
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

    with open(filename, encoding='ISO-8859-1') as f:
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


def generate_data(all_embeddings, neg_samples_file=''):
    """ Given a dictionary containing the embeddings of the sentences generate arrays of stories, correct-endings and
        wrong-endings. If the dataset doesn't contain wrong-endings negative samples (random or 1st nearest neighbour)
        are selected. For each ending e_i, we are computing its cosine distance with all the other endings of the training
        set |e|. The one that has the closest distance is assigned as wrong ending of the story.
        Parameters:
        -----------
        all_embeddings: dictionary
        dictionary of embeddings {key: value}

        neg_samples_file: string
        path to the file containing the 1st nearest neighbour of each embedding

        Returns:
        --------
        stories: list
        array with embeddings for the story

        correct_endings: list
        array with embeddings for the correct endings

        wrong_endings: list
        array with embeddings for the wrong endings

        """
    stories = []
    correct_endings = []
    wrong_endings = []

    if neg_samples_file:
        closest_ending_pairs = get_closest_pairs(neg_samples_file)

    for key, val in all_embeddings.items():
        if neg_samples_file and key in closest_ending_pairs:
            story = val[:4]  # list of 4 array of 4800 float
            correct_ending = val[4]  # true ending
            wrong_ending = all_embeddings[closest_ending_pairs[key]][4]
            stories.append(story)
            correct_endings.append(correct_ending)
            wrong_endings.append(wrong_ending)
        elif not neg_samples_file:
            story = val[:4]  # list of 4 array of 4800 float
            correct_ending = val[4]  # true ending
            if len(val) > 5:
                wrong_ending = val[5]
                wrong_endings.append(wrong_ending)
            else:
                all_keys = list(all_embeddings.keys())
                all_keys.remove(key)
                wrong_ending = all_embeddings[random.choice(all_keys)][4]
                wrong_endings.append(wrong_ending)
            stories.append(story)
            correct_endings.append(correct_ending)

    return np.asarray(stories), np.asarray(correct_endings), np.asarray(wrong_endings)


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


def write_results_to_csv(filename, stories, results):
    with open(filename, "w") as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(stories)):
            writer.writerow(stories[i].get_story_with_both_endings_as_list() + list(str(results[i])))














