import tensorflow as tf
import numpy as np

import utils
from classification_model import SentenceClozeTaskModel




#  Parameters

# Data loading parameters
tf.flags.DEFINE_float("val_sample_percentage", .01, "Percentage of the training data used for validation")
tf.flags.DEFINE_string("data_file_path", "data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv", "Path to the testing data")
tf.flags.DEFINE_string("path_to_embeddings", "./data/embeddings_test/", "Path to the embeddings")
tf.flags.DEFINE_string("path_to_embeddings_id", "./data/embeddings_test/id.txt", "Path to the embeddings id")
tf.flags.DEFINE_string("story_type", "last_sentence", "Story type: {no_context, last_sentence, plot (first 4 sentences), full (4 sentences + ending)}")
tf.flags.DEFINE_string("num_embeddings_per_story", 6, "The number of sentences in a story.")
tf.flags.DEFINE_string("embeddings_dim", 4800, "The dimension of the embeddings")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/checkpoints", "Checkpoint directory from training run")
# Model parameters

# Test parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("verbose_for_debugging", True, "Allow info to be printed to understand the behaviour of the network")
tf.flags.DEFINE_boolean("verbose_for_experiments", True, "Print only the perplexity")

FLAGS = tf.flags.FLAGS

# Prepare data

# Load data
# Prepare the data
print("Load list of sentences \n")
story_embeddings, embeddings_id = utils.load_embeddings(FLAGS.path_to_embeddings, FLAGS.path_to_embeddings_id, FLAGS.num_embeddings_per_story, FLAGS.embeddings_dim)
print("Loading and preprocessing test datasets \n")
x_beginning_of_story_embeddings,  x_right_ending_embeddings, x_wrong_ending_embeddings, y_right_labels, y_wrong_labels = utils.generate_test_data(story_embeddings, FLAGS.story_type)

## EVALUATION ##

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        inputs_beginning = graph.get_operation_by_name("inputs_beginning").outputs[0]
        inputs_ending = graph.get_operation_by_name("inputs_ending").outputs[0]
        labels = graph.get_operation_by_name("labels").outputs[0]

        # Tensor we want to evaluate
        predictions = graph.get_operation_by_name("accuracy/predictions").outputs[0]
        probabilities = graph.get_operation_by_name("probabilites").outputs[0]
        # Generate batches for one epoch
        batches = utils.batch_iter(
            list(zip(x_beginning_of_story_embeddings, x_right_ending_embeddings, x_wrong_ending_embeddings, y_right_labels, y_wrong_labels)),
            FLAGS.batch_size,
            1,
            shuffle=False)

        if FLAGS.verbose_for_experiments:
            print("verbose_for_experiments: Only the perplexity will be shown for each sentence")

        for batch_id, batch in enumerate(batches):
            x_beginning_batch, x_right_ending_batch, x_wrong_ending_batch, y_right_batch, y_wrong_batch = zip(*batch)
            right_batch_predictions, right_batch_probabilities = sess.run(
                predictions,
                probabilities,
                {inputs_beginning: x_beginning_batch,
                 inputs_ending: x_right_ending_batch,
                 labels: y_right_batch}
            )

            wrong_batch_predictions, wrong_batch_probabilities = sess.run(
                predictions,
                probabilities,
                {inputs_beginning: x_beginning_batch,
                 inputs_ending: x_wrong_ending_batch,
                 labels: y_wrong_batch}
            )

            prediction_sentence = ''
            ground_truth_sentence = ''
            y_batch = y_batch[0]
            for i in range(len(right_batch_predictions[0])):
                #TO DO



