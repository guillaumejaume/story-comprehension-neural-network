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

        # first ending place holder dim = [batch_size x emd_dim]
        self.first_endings = tf.placeholder(dtype=tf.float32,
                                            shape=[None, self.embed_size],
                                            name='first_endings')

        # second ending place holder dim = [batch_size x emd_dim]
        self.second_endings = tf.placeholder(dtype=tf.float32,
                                             shape=[None, self.embed_size],
                                             name='second_endings')

        # labels place holder dim = [batch_size]
        self.labels = tf.placeholder(dtype=tf.int32,
                                     shape=None,
                                     name='labels')

        with tf.device('/gpu:0'):
            with tf.variable_scope("story_embedding"):

                self.sentence_weights = tf.get_variable("sentence_weights",
                                                        shape=[4, 1],
                                                        dtype=tf.float32,
                                                        initializer=tf.contrib.layers.xavier_initializer())

                stories = tf.reshape(self.stories, shape=(-1, 4))

                # dim = [batch_size x emd_dim]
                self.embedded_story = tf.matmul(stories, self.sentence_weights)
                self.embedded_story = tf.reshape(self.embedded_story, shape=(-1, embed_size))

            with tf.variable_scope("relational_network", reuse=tf.AUTO_REUSE):

                # relation between story embedding and first ending
                self.r_s1 = self.relational_network(use_first_ending=True)

                # relation between story embedding and second ending
                self.r_s2 = self.relational_network(use_first_ending=False)

            with tf.variable_scope("sigma"):

                # concat relations Story-First and Story-Second
                self.r_concat = tf.concat([self.r_s1, self.r_s2], axis=1)

                # 2 MLP layers with ReLu activation
                dense_1 = tf.contrib.layers.fully_connected(self.r_concat, 1200, scope='sigma_1')
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

    def relational_network(self, use_first_ending):
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
        if use_first_ending:
            x = tf.concat([self.embedded_story, self.first_endings], axis=1)
        else:
            x = tf.concat([self.embedded_story, self.second_endings], axis=1)

        dense_1 = tf.contrib.layers.fully_connected(x, 4800, scope='f_1')
        dense_2 = tf.contrib.layers.fully_connected(dense_1, 2400, scope='f_2')
        dense_3 = tf.contrib.layers.fully_connected(dense_2, 2400, scope='f_3', activation_fn=None)

        return dense_3
