import os
import time
import datetime
import numpy as np

import tensorflow as tf

import utils
from classification_model import ClassificationModel


# Data loading parameters
tf.flags.DEFINE_string("training_embeddings_dir", "./data/embeddings_training/", "Path to the embeddings used for training")
tf.flags.DEFINE_string("validation_embeddings_dir", "./data/embeddings_validation/", "Path to the embeddings used for validation")
tf.flags.DEFINE_float("percentage_of_val", 0.50, "Percentage of the val set added to training")

# Model parameters
tf.flags.DEFINE_integer("embedding_dim", 4800, "The dimension of the embeddings")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on validation set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# Prepare the data
print("Load training and validation embeddings \n")

# load training embeddings
all_training_embeddings = utils.load_embeddings(FLAGS.training_embeddings_dir,
                                                FLAGS.embedding_dim)

# load validation embeddings
all_validation_embeddings = utils.load_embeddings(FLAGS.validation_embeddings_dir,
                                                  FLAGS.embedding_dim)

# generate training data as np array
training_stories, training_true_endings, training_wrong_endings = utils.generate_data(all_training_embeddings,
                                                                                      generate_random_ending=True)
training_stories = training_stories + training_stories
training_endings = training_true_endings + training_wrong_endings
training_labels = [1] * len(training_true_endings) + [0] * len(training_wrong_endings)
print(training_labels)
training_true_endings = []
training_wrong_endings = []

# generate val data as np array
val_stories, val_true_endings, val_wrong_endings = utils.generate_data(all_validation_embeddings,
                                                                       generate_random_ending=False)
val_stories = val_stories + val_stories
val_endings = val_true_endings + val_wrong_endings
val_labels = [1] * len(val_true_endings) + [0] * len(val_wrong_endings)
val_true_endings = []
val_wrong_endings = []

#Split training data in validation and training
"""
train_sample_index = int(0.99 * float(len(training_stories)))
training_stories, val_stories = training_stories[:train_sample_index], training_stories[train_sample_index:]
training_endings, val_endings = training_endings[:train_sample_index], training_endings[train_sample_index:]
training_labels, val_labels = training_labels[:train_sample_index], training_labels[train_sample_index:]
"""

# summary
print('# of training stories: ', len(training_stories), ' with shape: ', np.shape(training_stories))
print('# of training end: ', len(training_endings), ' with shape: ', np.shape(training_endings))
print('# of training labels: ', len(training_labels), ' with shape: ', np.shape(training_labels))
print('\n')

print('# of val stories: ', len(val_stories), ' with shape: ', np.shape(val_stories))
print('# of val end: ', len(val_endings), ' with shape: ', np.shape(val_endings))
print('# of val labels: ', len(val_labels), ' with shape: ', np.shape(val_labels))
print('\n')

# generate training batches
batches = utils.batch_iter(list(zip(training_stories, training_endings, training_labels)),
                           FLAGS.batch_size,
                           FLAGS.num_epochs)

print("Loading and preprocessing done \n")

# Define the model and start training
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Initialize model
        model = ClassificationModel()

        # Training step
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # Define Adam optimizer
        learning_rate = 0.002
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(model.loss, global_step=global_step)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
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
        def train_step(story, ending, labels):
            """
            A single training step
            """
            feed_dict = {
                model.stories: story,
                model.endings: ending,
                model.labels: labels,
            }
            _, step, summaries, loss, accuracy = sess.run([
                    train_op,
                    global_step,
                    train_summary_op,
                    model.loss,
                    model.accuracy
                ],
                feed_dict
            )
            time_str = datetime.datetime.now().isoformat()
            print('\n\n')
            print("{}: step {}, acc {:g}".format(time_str, step, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(story, ending, labels, writer=None):
            """
            Evaluates model on the validation set
            """
            feed_dict = {
                model.stories: story,
                model.endings: ending,
                model.labels: labels,
            }
            step, summaries, predictions, accuracy = sess.run([
                    global_step,
                    val_summary_op,
                    model.predictions,
                    model.accuracy
                ],
                feed_dict
            )
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, acc {:g}".format(time_str, step, accuracy))
            print('Predictions: ', predictions)
            if writer:
                writer.add_summary(summaries, step)

        # TRAINING LOOP
        for batch in batches:
            x_beginning_batch, x_ending_batch, y_batch = zip(*batch)
            train_step(x_beginning_batch, x_ending_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            print("current_step ", current_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_beginning_val, x_ending_val, y_val, writer=val_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))