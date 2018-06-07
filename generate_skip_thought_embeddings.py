import numpy as np
import skipthoughts
from story import Story
import struct
import csv


def load_and_process_text_data(filename, for_testing=False, is_labeled=True):
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

def generate_and_save_word_embeddings_for_sentences_text(input_file, embeddings_output_path, embeddings_id_output_file, for_testing = False, is_labeled = True):
    """ Generate the embeddings and save them in a different file
       Parameters:
       -----------
       input_file: string
       path to the file to load the data from

       embeddings_output_file_path: string
       path to the directory to save the embeddings in

       embeddings_id_output_file: string
       path to the file to save the embeddings id in
    """
    list_of_stories = load_and_process_text_data(input_file, for_testing, is_labeled)
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)

    fout_id = open(embeddings_id_output_file, "wb")
    for story in list_of_stories:
        print(story)
        if not for_testing:
            embeddings = encoder.encode(story.get_story_with_right_ending_as_list())
        else:
            embeddings = encoder.encode(story.get_story_with_both_endings_as_list())
        output_file = open(embeddings_output_path + story.id, "wb")
        for embed in embeddings:
            b = bytes()
            b = b.join((struct.pack('f', e) for e in embed))
            output_file.write(b)
        print(story.id)
        output_file.close()
        fout_id.write(story.id)
        fout_id.write("\n")
    fout_id.close()


generate_and_save_word_embeddings_for_sentences_text("./data/test_nlu18.csv","./data/embeddings_test_eth_test/", "./data/embeddings_test_eth_test/id.txt", True, False)

