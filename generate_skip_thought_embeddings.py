import numpy as np
import skipthoughts
import utils
from story import Story
import struct

def generate_and_save_word_embeddings_for_sentences_text(input_file, embeddings_output_path, embeddings_id_output_file):
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
    list_of_stories = utils.load_and_process_text_data(input_file)
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)

    fout_id = open(embeddings_id_output_file, "wb")
    for story in list_of_stories:
        embeddings = encoder.encode(story.get_story_as_list())

        for i, embed in enumerate(embeddings):
            b = bytes()
            b = b.join((struct.pack('f', e) for e in embed))
            print embed
            output_file = open(embeddings_output_path + story.id + str(i), "wb")
            output_file.write(b)
            output_file.close()
        fout_id.write(story.id)
        fout_id.write("\n")
    fout_id.close()


generate_and_save_word_embeddings_for_sentences_text("./data/train_stories.csv","./data/embeddings/", "./data/embeddings/id.txt")
