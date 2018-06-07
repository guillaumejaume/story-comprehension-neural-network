import tensorflow as tf
import numpy as np

import utils

#  Parameters

# Data loading parameters
tf.flags.DEFINE_string("testing_embeddings_dir", "./data/embeddings_test/", "Path to the embeddings used for testing")

# Model parameters
tf.flags.DEFINE_integer("embedding_dim", 4800, "The dimension of the embeddings")

# Testing parameters
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1528219258/checkpoints", "Checkpoint directory from training run")
FLAGS = tf.flags.FLAGS

# Prepare data

# Load data
# load validation embeddings
all_testing_embeddings = utils.load_embeddings(FLAGS.testing_embeddings_dir,
                                               FLAGS.embedding_dim)
# generate data
test_stories, test_true_endings, test_wrong_endings = utils.generate_data(all_testing_embeddings)

# construct test input
test_labels = np.random.choice([0, 1], size=len(test_stories), p=[0.5, 0.5])
first_test_endings = []
second_test_endings = []

for i, idx in enumerate(test_labels):
    if idx == 0:
        first_test_endings.append(test_true_endings[i])
        second_test_endings.append(test_wrong_endings[i])
    else:
        first_test_endings.append(test_wrong_endings[i])
        second_test_endings.append(test_true_endings[i])
first_test_endings = np.asarray(first_test_endings)
second_test_endings = np.asarray(second_test_endings)


## EVALUATION ##

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        stories_ph = graph.get_operation_by_name("stories").outputs[0]
        first_endings_ph = graph.get_operation_by_name("first_endings").outputs[0]
        second_endings_ph = graph.get_operation_by_name("second_endings").outputs[0]
        labels_ph = graph.get_operation_by_name("labels").outputs[0]

        # Tensor we want to evaluate
        predictions_ph = graph.get_operation_by_name("accuracy/predictions").outputs[0]
        accuracy_ph = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        predictions, accuracy = sess.run([predictions_ph, accuracy_ph],
                                         {stories_ph: test_stories,
                                          first_endings_ph: first_test_endings,
                                          second_endings_ph: second_test_endings,
                                          labels_ph: test_labels})

        print('Accuracy: ', accuracy)
