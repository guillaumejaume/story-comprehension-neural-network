import numpy as np
import skipthoughts
import utils
from story import Story
import struct

def generate_and_save_word_embeddings_for_sentences_text(input_file, embeddings_output_path, embeddings_id_output_file, for_testing = False):
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
    list_of_stories = utils.load_and_process_text_data(input_file, for_testing)
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)

    print("len(list_of_stories) ",len(list_of_stories))
    fout_id = open(embeddings_id_output_file, "wb")
    for story in list_of_stories:
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


#generate_and_save_word_embeddings_for_sentences_text("./data/train_stories.csv","./data/embeddings/", "./data/embeddings/id.txt")
generate_and_save_word_embeddings_for_sentences_text("./data/test_validation.csv","./data/embeddings_test/", "./data/embeddings_test/id.txt", True)
