import tensorflow as tf


class RelationalModel:
    """
    New contributions:
        - Weighted sum of each sentence embedding to construct a story embedding
        - Move from a classification task to a "Find the most likely option between 2 options"

    """

    def __init__(self, embed_size):

        self.embed_size = embed_size

        # story place holder dim = [batch_size x #sent x emd_dim]
        self.stories = tf.placeholder(dtype=tf.float32,
                                      shape=[None, 4, self.embed_size],
                                      name='stories')

        # ending place holder dim = [batch_size x emd_dim]
        self.endings = tf.placeholder(dtype=tf.float32,
                                      shape=[None, self.embed_size],
                                      name='endings')

        # labels place holder dim = [batch_size]
        self.labels = tf.placeholder(dtype=tf.int32,
                                     shape=None,
                                     name='labels')

        with tf.device('/gpu:0'):

            with tf.variable_scope("relational_network", reuse=tf.AUTO_REUSE):

                self.r_1 = self.relational_network(self.stories[:, 0, :], self.first_endings)
                self.r_2 = self.relational_network(self.stories[:, 1, :], self.first_endings)
                self.r_3 = self.relational_network(self.stories[:, 2, :], self.first_endings)
                self.r_4 = self.relational_network(self.stories[:, 3, :], self.first_endings)

                self.r = self.r_1 + self.r_2 + self.r_3 + self.r_4

            with tf.variable_scope("sigma"):

                # 2 MLP layers with ReLu activation
                dense_1 = tf.contrib.layers.fully_connected(self.r, 1200, scope='sigma_1')
                dense_2 = tf.contrib.layers.fully_connected(dense_1, 1200, scope='sigma_2')

            with tf.variable_scope("softmax"):

                # softmax layer
                self.logits = tf.contrib.layers.fully_connected(dense_2, 2, scope='sigma_3', activation_fn=None)
                self.probabilities = tf.nn.softmax(self.logits, name='probabilities')

            with tf.variable_scope("loss"):

                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                           logits=self.logits)
                self.loss = tf.reduce_mean(self.loss)

            with tf.variable_scope("accuracy"):

                self.predictions = tf.argmax(self.probabilities,
                                             axis=1,
                                             name='predictions',
                                             output_type=tf.int32)

                self.is_equal = tf.equal(self.predictions, self.labels)
                self.accuracy = tf.reduce_mean(tf.cast(self.is_equal, tf.float32),
                                               name='accuracy')

    def relational_network(self, object_1, object_2):
        """ Relational network: 3 dense layers
            - ReLU activation for dense 1 & 2
            - Linear activation for dense 3
              Parameters:
              -----------
              - self: this
              - use_true: boolean
                if True -> use first ending for similarity
                else -> use second ending

              Returns:
              ----------
              r: similarity between the story and the ending
        """
        # if use_first_ending:
        #     x = tf.concat([self.embedded_story, self.first_endings], axis=1)
        # else:
        #     x = tf.concat([self.embedded_story, self.second_endings], axis=1)

        x = tf.concat([object_1, object_2], axis=1)

        dense_1 = tf.contrib.layers.fully_connected(x, 2400, scope='f_1')
        dense_2 = tf.contrib.layers.fully_connected(dense_1, 1200, scope='f_2', activation_fn=None)
        dense_3 = tf.contrib.layers.fully_connected(dense_2, 1200, scope='f_3', activation_fn=None)

        return dense_3