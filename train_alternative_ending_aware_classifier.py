import os
import time
import datetime
import numpy as np

import tensorflow as tf

import utils
from model_alternative_ending_aware_classifier import AlternativeEndingAwareClassifier


# Data loading parameters
tf.flags.DEFINE_string("training_embeddings_dir", "./data/embeddings_training/", "Path to the embeddings used for training")
tf.flags.DEFINE_string("validation_embeddings_dir", "./data/embeddings_validation/", "Path to the embeddings used for validation")
tf.flags.DEFINE_string("neg_samples_file", "./data/training_neighbours.txt", "Path to the file with the closest pairs")
tf.flags.DEFINE_boolean("use_val_for_training", False, "If true, use 90% val as training and 10% as val")

# Model parameters
tf.flags.DEFINE_integer("embedding_dim", 4800, "The dimension of the embeddings")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 20, "Evaluate model on validation set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 20, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")

FLAGS = tf.flags.FLAGS

# Prepare the data
print("Load training and validation embeddings \n")


if FLAGS.use_val_for_training:
    # load validation embeddings
    all_validation_embeddings = utils.load_embeddings(FLAGS.validation_embeddings_dir,
                                                      FLAGS.embedding_dim)

    # generate val data as np array
    val_stories, val_true_endings, val_wrong_endings = utils.generate_data(all_validation_embeddings)

    # balance val samples into training and validation
    train_sample_index = int(0.9 * float(len(val_stories)))
    training_stories, val_stories = val_stories[:train_sample_index], val_stories[train_sample_index:]
    training_true_endings, val_true_endings = val_true_endings[:train_sample_index], val_true_endings[train_sample_index:]
    training_wrong_endings, val_wrong_endings = val_wrong_endings[:train_sample_index], val_wrong_endings[train_sample_index:]

else:
    # load training embeddings
    all_training_embeddings = utils.load_embeddings(FLAGS.training_embeddings_dir,
                                                    FLAGS.embedding_dim)

    # generate training data as np array
    training_stories, training_true_endings, training_wrong_endings = utils.generate_data(all_training_embeddings,
                                                                                          FLAGS.neg_samples_file)

    # load validation embeddings
    all_validation_embeddings = utils.load_embeddings(FLAGS.validation_embeddings_dir,
                                                      FLAGS.embedding_dim)

    # generate val data as np array
    val_stories, val_true_endings, val_wrong_endings = utils.generate_data(all_validation_embeddings)

# summary
print('# of training stories: ', len(training_stories), ' with shape: ', np.shape(training_stories))
print('# of training true end: ', len(training_true_endings), ' with shape: ', np.shape(training_true_endings))
print('# of training wrong end: ', len(training_wrong_endings), ' with shape: ', np.shape(training_wrong_endings))
print('\n')

print('# of val stories: ', len(val_stories), ' with shape: ', np.shape(val_stories))
print('# of val true end: ', len(val_true_endings), ' with shape: ', np.shape(val_true_endings))
print('# of val wrong end: ', len(val_wrong_endings), ' with shape: ', np.shape(val_wrong_endings))
print('\n')

# construct TRAIN input
training_labels = np.random.choice([0, 1], size=len(training_stories), p=[0.5, 0.5])
first_training_endings = []
second_training_endings = []

for i, idx in enumerate(training_labels):
    if idx == 0:
        first_training_endings.append(training_true_endings[i])
        second_training_endings.append(training_wrong_endings[i])
    else:
        first_training_endings.append(training_wrong_endings[i])
        second_training_endings.append(training_true_endings[i])
first_training_endings = np.asarray(first_training_endings)
second_training_endings = np.asarray(second_training_endings)

# generate training batches
batches = utils.batch_iter(list(zip(training_stories, first_training_endings, second_training_endings, training_labels)),
                           FLAGS.batch_size,
                           FLAGS.num_epochs)

# construct VAL input
val_labels = np.random.choice([0, 1], size=len(val_stories), p=[0.5, 0.5])
first_val_endings = []
second_val_endings = []

for i, idx in enumerate(val_labels):
    if idx == 0:
        first_val_endings.append(val_true_endings[i])
        second_val_endings.append(val_wrong_endings[i])
    else:
        first_val_endings.append(val_wrong_endings[i])
        second_val_endings.append(val_true_endings[i])
first_val_endings = np.asarray(first_val_endings)
second_val_endings = np.asarray(second_val_endings)

print("Loading and preprocessing done \n")

# Define the model and start training
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Initialize model
        model = AlternativeEndingAwareClassifier(embed_size=FLAGS.embedding_dim)

        # Training step
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # Define Adam optimizer
        learning_rate = 0.00001
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
        def train_step(stories, first_endings, second_endings, labels):
            """
            A single training step
            """
            feed_dict = {
                model.stories: stories,
                model.first_endings: first_endings,
                model.second_endings: second_endings,
                model.labels: labels
            }
            _, step, summaries, loss, accuracy = sess.run([train_op,
                                                           global_step,
                                                           train_summary_op,
                                                           model.loss,
                                                           model.accuracy],
                                                          feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print('\n\n')
            print("{}: step {}, acc {:g}".format(time_str, step, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(stories, first_endings, second_endings, labels, writer=None):
            """
            Evaluates model on the validation set
            """
            feed_dict = {
                model.stories: stories,
                model.first_endings: first_endings,
                model.second_endings: second_endings,
                model.labels: labels
            }
            step, summaries, predictions, accuracy = sess.run([global_step,
                                                               val_summary_op,
                                                               model.predictions,
                                                               model.accuracy],
                                                              feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, acc {:g}".format(time_str, step, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # TRAINING LOOP
        for batch in batches:
            batch_stories, batch_first_endings, batch_second_endings, batch_labels = zip(*batch)
            train_step(batch_stories, batch_first_endings, batch_second_endings, batch_labels)
            current_step = tf.train.global_step(sess, global_step)
            print("current_step ", current_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(val_stories, first_val_endings, second_val_endings, val_labels, writer=val_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))