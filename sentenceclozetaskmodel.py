import tensorflow as tf


class SentenceClozeTaskModel:
    """
    Simple model with 3 fc hidden layers

    """

    def __init__(self):

        self.embed_size = 4800
        self.inputs_beginning = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.embed_size],
            name='inputs_beginning'
        )
        self.inputs_ending = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.embed_size],
            name='inputs_ending'
        )
        self.labels = tf.placeholder(
            dtype=tf.int32,
            shape=None,
            name='labels'
        )

        with tf.device('/gpu:0'):
            with tf.variable_scope("dense_layers"):
                self.inputs = self.inputs_beginning + self.inputs_ending

                dense_1 = tf.layers.dense(inputs=self.inputs, units=2400, activation=tf.nn.relu)
                dense_2 = tf.layers.dense(inputs=dense_1, units=1200, activation=tf.nn.relu)
                dense_3 = tf.layers.dense(inputs=dense_2, units=600, activation=tf.nn.relu)

                self.logits = tf.layers.dense(inputs=dense_3, units=2)
                self.probabilities = tf.nn.softmax(self.logits, name = 'probabilities')

            with tf.variable_scope("loss"):
                print('labels: ', self.labels.shape)
                print('logits: ', self.logits.shape)

                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                           logits=self.logits)
                self.loss = tf.reduce_mean(self.loss)

            with tf.variable_scope("accuracy"):

                print('proba: ', self.probabilities.shape)

                self.predictions = tf.argmax(
                    self.probabilities,
                    axis=1,
                    name='predictions',
                    output_type=tf.int32
                )

                self.is_equal = tf.equal(self.predictions, self.labels)
                self.accuracy = tf.reduce_mean(
                    tf.cast(self.is_equal, tf.float32),
                    name='accuracy'
                )


