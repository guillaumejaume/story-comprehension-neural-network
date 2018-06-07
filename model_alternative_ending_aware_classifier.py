import tensorflow as tf


class AlternativeEndingAwareClassifier:
    """
    AlternativeEndingAwareClassifier model:
        - Based on a Relation Network (RN)
        - Output a probability of option 1/2 being respectively the correct and wrong ending

    Class Parameters:
        - embed_size: embedding dimension

    TensorFlow Parameters:
        - stories: bs x 4 x emb_dim Tensor
        form the context story by stacking the embedding of the four intro sentences
        - first_endings :  bs x emb_dim Tensor
        one of the two alternative (can either be correct or wrong)
        - second_endings : bs x emd_dim Tensor
        the second alternative (can also either be correct or wrong)
        - labels: bs Tensor
        0 -> first ending is correct, 1 -> second ending is correct
    """

    def __init__(self, embed_size):

        # embedding dimension
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

            with tf.variable_scope("relational_network", reuse=tf.AUTO_REUSE):

                self.r_11 = self.relational_network(self.stories[:, 0, :], self.first_endings)
                self.r_12 = self.relational_network(self.stories[:, 1, :], self.first_endings)
                self.r_13 = self.relational_network(self.stories[:, 2, :], self.first_endings)
                self.r_14 = self.relational_network(self.stories[:, 3, :], self.first_endings)

                self.r_21 = self.relational_network(self.stories[:, 0, :], self.second_endings)
                self.r_22 = self.relational_network(self.stories[:, 1, :], self.second_endings)
                self.r_23 = self.relational_network(self.stories[:, 2, :], self.second_endings)
                self.r_24 = self.relational_network(self.stories[:, 3, :], self.second_endings)

                ######################################################################
                # Sum each relation
                self.r1 = self.r_11 + self.r_12 + self.r_13 + self.r_14
                self.r2 = self.r_21 + self.r_22 + self.r_23 + self.r_24
                ######################################################################

                ######################################################################
                # # Alternative relation aggregation: take the relation with max norm
                # r1 = tf.stack([self.r_11, self.r_12, self.r_13, self.r_14], axis=1)
                # norm_r1 = tf.norm(r1, axis=2)
                # index_max_r1 = tf.argmax(norm_r1, axis=1, output_type=tf.int32)
                #
                # self.max_indices = index_max_r1
                # self.norms = norm_r1
                #
                # r2 = tf.stack([self.r_21, self.r_22, self.r_23, self.r_24], axis=1)
                # norm_r2 = tf.norm(r1, axis=2)
                # index_max_r2 = tf.argmax(norm_r2, axis=1, output_type=tf.int32)
                #
                # self.r1 = self.extract_max_norm(r1, index_max_r1)
                # self.r2 = self.extract_max_norm(r2, index_max_r2)
                # ######################################################################

                # concat relation from r1 and r2
                self.r_concat = tf.concat([self.r1, self.r2], axis=1)

                # 2 MLP layers with ReLu activation
                dense_1 = tf.contrib.layers.fully_connected(self.r_concat, 1200, scope='sigma_1')
                dropout_1 = tf.layers.dropout(inputs=dense_1, rate=0.5)
                dense_2 = tf.contrib.layers.fully_connected(dropout_1, 1200, scope='sigma_2')
                dropout_2 = tf.layers.dropout(inputs=dense_2, rate=0.5)

            with tf.variable_scope("softmax"):

                # softmax layer
                self.logits = tf.contrib.layers.fully_connected(dropout_2, 2, scope='sigma_3', activation_fn=None)
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

    # def extract_max_norm(self, r, max_indices):
    #     """ Extract the max norm in the second dim of r
    #           Parameters:
    #           -----------
    #           - r: bs x 4 x emd_dim tensor
    #           - max_indices:
    #           index of the vector (out of the 4 in dim 2) with max norm
    #
    #           Returns:
    #           ----------
    #           out: bs x emd_dim tensor
    #     """
    #
    #     batch_size = tf.shape(r)[0]
    #     batch_id = tf.constant(0, dtype=tf.int32)
    #
    #     def condition(batch_id, output):
    #         return batch_id < batch_size
    #
    #     def process_batch_step(batch_id, output_ta_t):
    #         current_batch_input = input_as_ta.read(batch_id)
    #         new_output = current_batch_input[max_indices[batch_id], :]
    #         output_ta_t = output_ta_t.write(batch_id, new_output)
    #         return batch_id + 1, output_ta_t
    #
    #     output_as_ta = tf.TensorArray(size=batch_size, dtype=tf.float32)
    #     input_as_ta = tf.TensorArray(size=batch_size, dtype=tf.float32)
    #     input_as_ta = input_as_ta.unstack(r)
    #
    #     _, out = tf.while_loop(cond=condition,
    #                            body=process_batch_step,
    #                            loop_vars=(batch_id, output_as_ta))
    #
    #     out = out.stack()
    #     return out

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
        x = tf.concat([object_1, object_2], axis=1)

        dense_1 = tf.contrib.layers.fully_connected(x, 2400, scope='f_1')
        dense_2 = tf.contrib.layers.fully_connected(dense_1, 1200, scope='f_2')
        dense_3 = tf.contrib.layers.fully_connected(dense_2, 1200, scope='f_3', activation_fn=None)

        return dense_3
