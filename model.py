import tensorflow as tf


class Model:
    """
    @TODO describe briefly the method...
    """

    def __init__(self):

        # @TODO set placeholders

        with tf.device('/gpu:0'):

            with tf.variable_scope("loss"):

                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                           logits=self.logits)
                self.loss = tf.reduce_mean(self.loss)

            with tf.variable_scope("accuracy"):

                self.predictions = tf.argmax(self.probabilities,
                                             axis=2,
                                             output_type=tf.int32,
                                             name='predictions')
                self.is_equal = tf.equal(self.predictions, self.labels)
                self.accuracy = tf.reduce_mean(tf.cast(self.is_equal, tf.float32),
                                               name='accuracy')


