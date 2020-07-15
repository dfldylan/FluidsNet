import tensorflow as tf

class FeatureNet(object):
    def __init__(self, RANDOM_NUMBER, VOXEL_NUMBER, mini_max_particles_voxel, max_particles_voxel, grid_size):
        # train_input[N, 7] & [N, K, 7], train_output[N, 3], voxel_index[N]
        # input
        self.feature = tf.placeholder(tf.float32, [None, 7], "particle_feature")
        self.neighbor = tf.placeholder(tf.float32, [None, 27 * mini_max_particles_voxel, 7], "neighbor_feature")
        self.voxel_index = tf.placeholder(tf.int32, [VOXEL_NUMBER, max_particles_voxel], "voxel_index")
        # output
        self.truth = tf.placeholder(tf.float32, [None, 3])

        self.pred, self.sf = self.trans(RANDOM_NUMBER, VOXEL_NUMBER, max_particles_voxel, grid_size)  # [N, 3]

        loss_full = tf.abs(self.truth - self.pred)  # [N, 3]
        fluid_divide = tf.dynamic_partition(loss_full, self.sf, 2)

        self.loss = tf.reduce_sum(fluid_divide[0]) / tf.cast(tf.shape(fluid_divide[0])[0], tf.float32)

    def trans(self, RANDOM_NUMBER, VOXEL_NUMBER, max_particles_voxel, grid_size):
        fcn9 = tf.layers.dense(self.neighbor, 9, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer,
                               name='fcn9')  # [N, size, 9]
        mean9 = tf.reduce_mean(fcn9, axis=1, name="mean9")  # [B, 9]
        particles_with_knn = tf.concat([self.feature, mean9], axis=1)  # [B, 16]

        voxel_feature = tf.gather(
            tf.concat([tf.truncated_normal([RANDOM_NUMBER, 16], 0.5, 0.25), particles_with_knn], axis=0),
            self.voxel_index + tf.constant(RANDOM_NUMBER), name="voxel_divide")  # [21* 21* 21, 64, 16]

        voxel_feature = tf.reshape(voxel_feature, [VOXEL_NUMBER * max_particles_voxel, 1, 16])  # [21* 21* 21* 64, 1, 16]

        conv1 = tf.reshape(tf.layers.conv1d(voxel_feature, 32, [1], name="conv1d_1", activation=tf.nn.tanh,
                                      kernel_initializer=tf.random_normal_initializer,
                                      bias_initializer=tf.random_normal_initializer), [VOXEL_NUMBER, max_particles_voxel, 32])  # [21* 21* 21, 64, 32]
        max32 = tf.reduce_max(conv1, axis=1, keepdims=True, name="max32")  # [21* 21* 21, 1, 32]
        feature64 = tf.concat([conv1, tf.tile(max32, [1, max_particles_voxel, 1])], axis=2)  # [21* 21* 21, 64, 64]
        conv2 = tf.layers.conv1d(tf.reshape(feature64, [VOXEL_NUMBER * max_particles_voxel, 1, 64]), 128, [1], name="conv1d_2", activation=tf.nn.tanh,
                                      kernel_initializer=tf.random_normal_initializer,
                                      bias_initializer=tf.random_normal_initializer)  # [21* 21* 21* 64, 1, 128]
        max128 = tf.reduce_max(tf.reshape(conv2, [VOXEL_NUMBER, max_particles_voxel, 128]), axis=1, keepdims=False, name="max128")  # [21* 21* 21, 128]

        assert grid_size[0] == grid_size[1] == grid_size[2] == 21
        tensor_4D = tf.reshape(max128, [1, 21, 21, 21, 128])

        conv3 = tf.layers.conv3d(tensor_4D, 64, [2, 2, 2], [2, 2, 2], padding="same", name="conv3d_1", activation=tf.nn.tanh,
                                      kernel_initializer=tf.random_normal_initializer,
                                      bias_initializer=tf.random_normal_initializer)  # [1, 11, 11, 11, 64]
        conv4 = tf.layers.conv3d(conv3, 64, [2, 2, 2], [2, 2, 2], padding="same", name="conv3d_2", activation=tf.nn.tanh,
                                      kernel_initializer=tf.random_normal_initializer,
                                      bias_initializer=tf.random_normal_initializer)  # [1, 6, 6, 6, 64]
        conv5 = tf.layers.conv3d(conv4, 32, [2, 2, 2], [2, 2, 2], padding="same", name="conv3d_3", activation=tf.nn.tanh,
                                      kernel_initializer=tf.random_normal_initializer,
                                      bias_initializer=tf.random_normal_initializer)  # [1, 3, 3, 3, 32]
        conv6 = tf.layers.conv3d(conv5, 32, [3, 3, 3], [3, 3, 3], padding="same", name="conv3d_4", activation=tf.nn.tanh,
                                      kernel_initializer=tf.random_normal_initializer,
                                      bias_initializer=tf.random_normal_initializer)  # [1, 1, 1, 1, 32]
        dim = tf.shape(particles_with_knn)[0]
        final_feature = tf.concat([particles_with_knn, tf.tile(tf.reshape(conv6, [1, 32]), [dim, 1])],
                                       axis=1)  # [-1, 48]

        fcn1 = tf.layers.dense(final_feature, 32, name="fcn1", activation=tf.nn.tanh,
                                    kernel_initializer=tf.random_normal_initializer)  # [-1, 32]
        fcn2 = tf.layers.dense(fcn1, 16, name="fcn2", activation=tf.nn.tanh,
                                    kernel_initializer=tf.random_normal_initializer)  # [-1, 16]
        fcn3 = tf.layers.dense(fcn2, 8, name="fcn3", activation=tf.nn.tanh,
                                    kernel_initializer=tf.random_normal_initializer)  # [-1, 8]
        fcn4 = tf.layers.dense(fcn3, 3, name="fcn4", kernel_initializer=tf.random_normal_initializer)  # [-1, 3]
        final = fcn4 * tf.expand_dims(tf.ones([tf.shape(final_feature)[0]]) - final_feature[:, 6], axis=1)
        sf = tf.cast(final_feature[:, 6], tf.int32)

        return final, sf  # [N, 3]

class Model(object):
    def __init__(self, learning_rate, avail_gpus, mini_max_particles_voxel, max_particles_voxel, grid_size, VOXEL_NUMBER, RANDOM_NUMBER):
        self.learning_rate = learning_rate
        self.avail_gpus = avail_gpus
        self.global_step = tf.train.get_or_create_global_step()

        self.feature = []
        self.neighbor = []
        self.voxel_index = []
        self.labels = []
        self.loss = []
        self.pred = []

        with tf.variable_scope(tf.get_variable_scope()):
            for idx, dev in enumerate(self.avail_gpus):
                with tf.device('/gpu:{}'.format(dev)), tf.name_scope('gpu_{}'.format(dev)) as scope:
                    # must use name scope here since we do not want to create new variables
                    # graph
                    feature = FeatureNet(RANDOM_NUMBER, VOXEL_NUMBER, mini_max_particles_voxel, max_particles_voxel, grid_size)
                    tf.get_variable_scope().reuse_variables()
                    # input
                    self.feature.append(feature.feature)
                    self.neighbor.append(feature.neighbor)
                    self.voxel_index.append(feature.voxel_index)
                    self.labels.append(feature.truth)
                    self.loss.append(feature.loss)
                    self.pred.append(feature.pred)

        self.params = tf.trainable_variables()
        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.loss_final = tf.reduce_mean(self.loss)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimazer = self.opt.minimize(self.loss_final, global_step=self.global_step)

        # summary and saver
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                                    max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

        self.train_summary = tf.summary.merge([
            tf.summary.scalar('train/loss', self.loss_final),
            *[tf.summary.histogram(each.name, each) for each in self.vars + self.params]
        ])

    def train(self, session, data, train=False, summary=False, onlypred=False):
        input_feed = {}
        for idx in range(len(self.avail_gpus)):
            input_feed[self.voxel_index[idx]] = data[3][idx]  # [N]
            input_feed[self.feature[idx]] = data[0][idx]  # [N, 7]
            input_feed[self.neighbor[idx]] = data[1][idx]  # [N, K]
            input_feed[self.labels[idx]] = data[2][idx]  # [N, 3]

        output_feed = [self.pred, self.loss_final, self.global_step]
        if onlypred:
            output_feed = [self.pred]
        elif train:
            output_feed.append(self.optimazer)
            if summary:
                output_feed.append(self.train_summary)

        ret = session.run(output_feed, input_feed)
        # pdb.set_trace()
        return ret


