import utils

from model import Model

import tensorflow as tf
import os
import time
import datetime

import numpy as np

#Training parameters

# Data loading parameters
tf.flags.DEFINE_float("val_sample_percentage", .001, "Percentage of the training data used for validation")
tf.flags.DEFINE_string("data_file_path", "data/train_stories_small.csv", "Path to the training data")
tf.flags.DEFINE_string("path_to_embeddings", "data/embeddings_fin.txt", "Path to the embeddings")
tf.flags.DEFINE_string("path_to_embeddings_id", "data/embeddings_id.txt", "Path to the embeddings id")

# Model parameters
tf.flags.DEFINE_integer("embedding_dimension", 100, "Dimensionality of word embeddings")
tf.flags.DEFINE_integer("vocabulary_size", 20000, "Size of the vocabulary")
tf.flags.DEFINE_integer("state_size", 512, "Size of the hidden LSTM state")
tf.flags.DEFINE_integer("sentence_length", 30, "Length of each sentence fed to the LSTM")

# Embedding parameters
tf.flags.DEFINE_boolean("use_word2vec_emb", True, "Use word2vec embedding")
tf.flags.DEFINE_string("path_to_word2vec", "wordembeddings-dim100.word2vec", "Path to the embedding file")

# Training parameters
tf.flags.DEFINE_integer("max_grad_norm", 5, "max norm of the gradient")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on validation set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# for running on EULER, adapt this
tf.flags.DEFINE_integer("inter_op_parallelism_threads", 0,
                        "TF nodes that perform blocking operations are enqueued on a pool of "
                        "inter_op_parallelism_threads available in each process.")
tf.flags.DEFINE_integer("intra_op_parallelism_threads", 0,
                        "The execution of an individual op (for some op types) can be parallelized"
                        " on a pool of intra_op_parallelism_threads.")

FLAGS = tf.flags.FLAGS

# Prepare the data
print("Load list of sentences \n")
vocab = []
generated_embeddings = []

print("Loading and preprocessing training and validation datasets \n")
data = []
labels = []

# Randomly shuffle data
np.random.seed(10)
shuffled_indices = np.random.permutation(len(labels))
data = data[shuffled_indices]
labels = labels[shuffled_indices]

# Split train/dev sets
val_sample_index = -1 * int(FLAGS.val_sample_percentage * float(len(labels)))
x_train, x_val = data[:val_sample_index], data[val_sample_index:]
y_train, y_val = labels[:val_sample_index], labels[val_sample_index:]

# Summary of the loaded data
print('Loaded: ', len(x_train), ' samples for training')
print('Loaded: ', len(x_val), ' samples for validation')

print('Training input has shape: ', np.shape(x_train))
print('Validation input has shape: ', np.shape(x_val))

print('Training labels has shape: ', np.shape(y_train))
print('Validation labels has shape: ', np.shape(y_val))

# Generate training batches
batches = utils.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

print("Loading and preprocessing done \n")

# Define the model and start training
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
        intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Initialize model
        model = Model()

        # Training step
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # Define Adam optimizer
        learning_rate = 0.0002
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize()

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss, perplexity and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Validation summaries
        val_summary_op = tf.summary.merge([loss_summary, acc_summary])
        val_summary_dir = os.path.join(out_dir, "summaries", "dev")
        val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

        # Checkpoint directory (Tensorflow assumes this directory already exists so we need to create it)
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()

        # Define training and validation steps (batch)
        def train_step(inputs, labels, vocab_emb):
            """
            A single training step
            """
            feed_dict = {
                model.inputs: inputs,
                model.labels: labels,
                model.vocab_embedding: vocab_emb,
                model.discard_last_prediction: True
            }
            _, step, summaries, loss, perplexity, accuracy = sess.run([train_op,
                                                                       global_step,
                                                                       train_summary_op,
                                                                       model.loss,
                                                                       model.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print('\n\n')
            print("{}: step {}, perplexity {:g}, acc {:g}".format(time_str, step, perplexity, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(inputs, labels, vocab_emb, writer=None):
            """
            Evaluates model on the validation set
            """
            feed_dict = {
                model.inputs: inputs,
                model.labels: labels,
                model.vocab_embedding: vocab_emb,
                model.discard_last_prediction: True
            }
            step, summaries, predictions, perplexity, accuracy = sess.run([global_step,
                                                                           val_summary_op,
                                                                           model.predictions,
                                                                           model.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, perplexity {:g}, acc {:g}".format(time_str, step, perplexity, accuracy))
            print('Predictions: ', predictions)
            if writer:
                writer.add_summary(summaries, step)

        # TRAINING LOOP
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch, vocab_emb)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_val, y_val, vocab_emb, writer=val_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
