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

                init = tf.constant_initializer([0.25, 0.25, 0.25, 0.25])
                self.sentence_weights = tf.get_variable("sentence_weights",
                                                        shape=[4, 1],
                                                        dtype=tf.float32,
                                                        initializer=init,
                                                        constraint=tf.keras.constraints.min_max_norm(min_value=1,
                                                                                                     max_value=1))

                # # self.sentence_weights = tf.clip_by_value(self.sentence_weights,
                # #                                          clip_value_min=0,
                # #                                          clip_value_max=1,
                # #                                          name='clip_weights')
                #
                # stories = tf.reshape(self.stories, shape=(-1, 4))
                #
                # # dim = [batch_size x emd_dim]
                # self.embedded_story = tf.matmul(stories, self.sentence_weights)
                # self.embedded_story = tf.reshape(self.embedded_story, shape=(-1, embed_size))

                # EXP: assign 0 weight to ALL the sentences except the last one
                # self.embedded_story = self.stories[:, -1, :]

                # EXP: sum al the vectors
                # self.embedded_story = self.stories[:, 0, :] + self.stories[:, 1, :] +
                #                       self.stories[:, 2, :] + self.stories[:, 3, :]

            with tf.variable_scope("relational_network", reuse=tf.AUTO_REUSE):

                self.r_11 = self.relational_network(self.stories[:, 0, :], self.first_endings)
                self.r_12 = self.relational_network(self.stories[:, 1, :], self.first_endings)
                self.r_13 = self.relational_network(self.stories[:, 2, :], self.first_endings)
                self.r_14 = self.relational_network(self.stories[:, 3, :], self.first_endings)

                self.r_21 = self.relational_network(self.stories[:, 0, :], self.second_endings)
                self.r_22 = self.relational_network(self.stories[:, 1, :], self.second_endings)
                self.r_23 = self.relational_network(self.stories[:, 2, :], self.second_endings)
                self.r_24 = self.relational_network(self.stories[:, 3, :], self.second_endings)

                self.r1 = self.r_11 + self.r_12 + self.r_13 + self.r_14
                self.r2 = self.r_21 + self.r_22 + self.r_23 + self.r_24

                # # relation between story embedding and first ending
                # self.r_s1 = self.relational_network(self.embedded_story, self.first_endings)
                #
                # # relation between story embedding and second ending
                # self.r_s2 = self.relational_network(self.embedded_story, self.second_endings)
                #
                # # relation between first ending and second ending
                # self.r_s3 = self.relational_network(self.first_endings, self.second_endings)

            with tf.variable_scope("sigma"):

                # concat relations Story-First and Story-Second
                # self.r_concat = tf.concat([self.r_s1, self.r_s2], axis=1)
                # self.r_concat = tf.concat([self.r_11, self.r_12, self.r_13, self.r_14,
                #                            self.r_21, self.r_22, self.r_23, self.r_24], axis=1)

                self.r_concat = tf.concat([self.r1, self.r2], axis=1)

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
