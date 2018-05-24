import utils

from sentenceclozetaskmodel import SentenceClozeTaskModel

import tensorflow as tf
import os
import time
import datetime
import random
import numpy as np

#Training parameters

# Data loading parameters
tf.flags.DEFINE_float("val_sample_percentage", .01, "Percentage of the training data used for validation")
tf.flags.DEFINE_string("data_file_path", "data/train_stories.csv", "Path to the training data")
tf.flags.DEFINE_string("path_to_embeddings", "./data/embeddings/", "Path to the embeddings")
tf.flags.DEFINE_string("path_to_embeddings_id", "./data/embeddings/id.txt", "Path to the embeddings id")
tf.flags.DEFINE_string("story_type", "last_sentence", "Story type: {no_context, last_sentence, plot (first 4 sentences), full (4 sentences + ending)}")
tf.flags.DEFINE_string("num_embeddings_per_story", 5, "The number of sentences in a story.")
tf.flags.DEFINE_string("embeddings_dim", 4800, "The dimension of the embeddings")
tf.flags.DEFINE_string("generate_radom_ending", True, "Generate random ending for the dataset that lacks it (eg. training dataset)")

# Model parameters

# Embedding parameters

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
story_embeddings, embeddings_id = utils.load_embeddings(FLAGS.path_to_embeddings, FLAGS.path_to_embeddings_id, FLAGS.num_embeddings_per_story, FLAGS.embeddings_dim)
print("Loading and preprocessing training and validation datasets \n")
beginning_of_story_embeddings,  ending_embeddings, labels = utils.generate_training_data(story_embeddings, FLAGS.story_type, FLAGS.generate_random_ending)

# Randomly shuffle data
data = list(zip(beginning_of_story_embeddings, ending_embeddings, labels))
random.shuffle(data)
beginning_of_story_embeddings, ending_embeddings, labels = zip(*data)

print(len(labels), "labels")
print(len(beginning_of_story_embeddings), "bose")
print(len(ending_embeddings), "ee")

# Split train/dev sets
val_sample_index = -1 * int(FLAGS.val_sample_percentage * float(len(labels)))
val_sample_index = 5
x_beginning_train, x_beginning_val = beginning_of_story_embeddings[:val_sample_index], beginning_of_story_embeddings[val_sample_index:]
x_ending_train, x_ending_val = ending_embeddings[:val_sample_index], ending_embeddings[val_sample_index:]
y_train, y_val = labels[:val_sample_index], labels[val_sample_index:]

# Summary of the loaded data
print('Loaded: ', len(x_beginning_train), ' samples for training')
print('Loaded: ', len(x_beginning_val), ' samples for validation')

print('Training input1 has shape: ', np.shape(x_beginning_train))
print('Validation input1 has shape: ', np.shape(x_beginning_val))

print('Training input2 has shape: ', np.shape(x_ending_train))
print('Validation input2 has shape: ', np.shape(x_ending_val))

print('Training labels has shape: ', np.shape(y_train))
print('Validation labels has shape: ', np.shape(y_val))

# Generate training batches
batches = utils.batch_iter(list(zip(x_beginning_train, x_ending_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

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
        model = SentenceClozeTaskModel()

        # Training step
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # Define Adam optimizer
        learning_rate = 0.0002
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
        def train_step(inputs_beginning, inputs_ending, labels):
            """
            A single training step
            """
            feed_dict = {
                model.inputs_beginning: inputs_beginning,
                model.inputs_ending: inputs_ending,
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

        def dev_step(inputs_beginning, inputs_ending, labels, writer=None):
            """
            Evaluates model on the validation set
            """
            feed_dict = {
                model.inputs_beginning: inputs_beginning,
                model.inputs_ending: inputs_ending,
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