import tensorflow as tf
import numpy as np

import utils


# Data loading parameters
tf.flags.DEFINE_string("testing_embeddings_dir", "./data/embeddings_test_eth/", "Path to the embeddings used for testing")
tf.flags.DEFINE_string("testing_stories", "./data/test_nlu18.csv", "Path to the file with the stories.")

# Model parameters
tf.flags.DEFINE_integer("embedding_dim", 4800, "The dimension of the embeddings")

# Testing parameters
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1528379186/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_string("output_file", "./output.csv", "Csv file containing the results")
tf.flags.DEFINE_boolean("has_labels", True, "if has_labels => compute accuracy, if not dump output in file")

FLAGS = tf.flags.FLAGS

# load testing embeddings
all_testing_embeddings = utils.load_embeddings(FLAGS.testing_embeddings_dir,
                                               FLAGS.embedding_dim)

# generate data
test_stories, test_true_endings, test_wrong_endings = utils.generate_data(all_testing_embeddings)
test_stories = np.concatenate((test_stories, test_stories),  axis=0)
test_endings = np.concatenate((test_true_endings, test_wrong_endings), axis=0)

# construct test input
test_labels = [1] * len(test_true_endings) + [0] * len(test_wrong_endings)

## EVALUATION ##

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
                      allow_soft_placement=True,
                      log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
      
        # Get the placeholders from the graph by name
        stories_ph = graph.get_operation_by_name("stories").outputs[0]
        endings_ph = graph.get_operation_by_name("endings").outputs[0]
        
        # Tensor we want to evaluate
        predictions_ph = graph.get_operation_by_name("accuracy/predictions").outputs[0]
        probabilities_ph = graph.get_operation_by_name("softmax/probabilities").outputs[0]

        if FLAGS.has_labels:
            labels_ph = graph.get_operation_by_name("labels").outputs[0]
            accuracy_ph = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
            predictions, accuracy, probabilities = sess.run([
                    predictions_ph, accuracy_ph, probabilities_ph
                ], {
                    stories_ph: test_stories,
                    endings_ph: test_endings,
                    labels_ph: test_labels
                })

            slice_index = int(len(probabilities)/2)
            prob = np.concatenate((probabilities[slice_index:,:], probabilities[:slice_index,:]), axis = 1)
            res = [int(prob[i][0] > prob[i][2]) for i in range(len(prob))]
            accuracy = sum(res)/len(res)
  
            print('Accuracy: ', accuracy)
        else:
            predictions, probabilities = sess.run([
                    predictions_ph, probabilities_ph
                ], {
                    stories_ph: test_stories,
                    endings_ph: test_endings
                })
            slice_index = int(len(probabilities) / 2)
            prob = np.concatenate((probabilities[slice_index:, :], probabilities[:slice_index, :]), axis=1)
            res = [int(prob[i][0] > prob[i][2]) for i in range(len(prob))]
            res = [r + 1 for r in res]
            all_testing_stories = utils.load_and_process_text_data(
                FLAGS.testing_stories, for_testing=True,
                is_labeled=FLAGS.has_labels)
            utils.write_results_to_csv(FLAGS.output_file, all_testing_stories, res)
