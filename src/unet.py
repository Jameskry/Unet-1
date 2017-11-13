import tensorflow as tf
import numpy as np
import logging
from layers import *
from collections import OrderedDict
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class HyperParams(object):
    def __init__(self, model_path, log_path, result_path, channels=1, batch_size=64, filter_size=3, pool_size=2,
                 layer_num=5, init_feature_num=64, class_num=2, class_weights=[1, 1], loss_type='cross_entropy',
                 optimizer_type='adam', learning_rate=0.0001, decay_rate=0.5, momentum=0.9, train_iters=100000,
                 epochs=50, report_step=100, need_summary=True, need_restore=False, need_eval=True):
        ''' data '''
        self.batch_size = batch_size
        self.channels = channels
        ''' structure '''
        self.filter_size = filter_size
        self.layer_num = layer_num
        self.init_feature_num = init_feature_num
        self.class_num = class_num
        self.pool_size = pool_size
        ''' loss '''
        self.loss_type = loss_type
        self.class_weights = class_weights
        ''' options'''
        self.need_summary = need_summary
        self.need_restore = need_restore
        self.need_eval = need_eval
        ''' optimizer'''
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.momentum = momentum
        '''training'''
        self.train_iters = train_iters
        self.epochs = epochs
        self.report_step = report_step
        '''path'''
        self.model_path = model_path
        self.log_path = log_path
        self.result_path = result_path

    def to_string(self):
        return 'height:' + str(self.height)



class Unet(object):
    def __init__(self, hps):
        ''' initialize the network '''

        ''' hyperparameters '''
        self.hps = hps
        self.learning_rate = tf.Variable(self.hps.learning_rate)
        self.global_step = tf.Variable(0)

        ''' input '''
        self.x = tf.placeholder(tf.float32, shape=[None, None, None, self.hps.channels])
        self.y = tf.placeholder(tf.uint8, shape=[None, None, 1])
        self.y_one_hot = tf.one_hot(indices=self.y, depth=self.hps.class_num)

        ''' tensors '''
        self.logits = self._build_graph()
        self.prediction = self._get_prediction()
        self.loss = self._get_loss()
        self.accuracy = self._get_accuracy()
        self.dice = self._get_dice()
        self.train_op = self._get_train_op()
        self.summary_op = self._get_summary_op()

    def _build_graph(self):
        ''' build the logits '''

        ''' initialize input '''
        input_x = tf.reshape(self.x, [tf.hps.batch_size, tf.shape(self.x)[1], tf.shape(self.x)[2], self.hps.channels])

        self.convs = []
        self.pooling_layers = OrderedDict()
        self.deconv_layers = OrderedDict()
        self.encoding_conv_layers = OrderedDict()
        self.decoding_conv_layers = OrderedDict()

        ''' encoding '''
        for layer_index in range(0, self.hps.layer_num):
            features = (2 ** layer_index) * self.hps.init_feature_num
            stddev = np.sqrt(2 / (self.hps.filter_size ** 2 * features))
            if layer_index == 0:
                w1 = weight_variable([self.hps.filter_size, self.hps.filter_size, self.hps.channels, features])
            else:
                w1 = weight_variable([self.hps.filter_size, self.hps.filter_size, features // 2, features], stddev)
            w2 = weight_variable([self.hps.filter_size, self.hps.filter_size, features, features], stddev)
            b1 = bias_variable([features])
            b2 = bias_variable([features])

            ''' add convolution layers '''
            # conv1 = conv2d(input_x, w1, keep_prob)
            conv1 = conv2d(input_x, w1, tf.constant(1.0))
            tmp_h_conv = tf.nn.relu(conv1 + b1)
            # conv2 = conv2d(tmp_h_conv, w2, keep_prob)
            conv2 = conv2d(tmp_h_conv, w2, tf.constant(1.0))
            self.encoding_conv_layers[layer_index] = tf.nn.relu(conv2 + b2)

            self.convs.append((conv1, conv2))

            ''' add pooling layers '''
            if layer_index < self.hps.layer_num - 1:
                self.pooling_layers[layer_index] = max_pool(self.encoding_conv_layers[layer_index], self.hps.pool_size)
                input_x = self.pooling_layers[layer_index]

        input_x = self.encoding_conv_layers[self.hps.layer_num - 1]

        ''' decoding '''
        for layer_index in range(self.hps.layer_num - 2, -1, -1):
            features = 2 ** (layer_index + 1) * self.hps.init_feature_num
            stddev = np.sqrt(2 / (self.hps.filter_size ** 2 * features))
            wd = weight_variable_devonc([self.hps.pool_size, self.hps.pool_size, features // 2, features], stddev)
            bd = bias_variable([features // 2])
            ''' add deconv layers'''
            deconv = tf.nn.relu(deconv2d(input_x, wd, self.hps.pool_size) + bd)
            ''' short cut '''
            deconv_concat = concat(self.encoding_conv_layers[layer_index], deconv)
            self.deconv_layers[layer_index] = deconv_concat

            w1 = weight_variable([self.hps.filter_size, self.hps.filter_size, features, features // 2], stddev)
            w2 = weight_variable([self.hps.filter_size, self.hps.filter_size, features // 2, features // 2], stddev)
            b1 = bias_variable([features // 2])
            b2 = bias_variable([features // 2])

            ''' add conv layers'''
            # conv1 = conv2d(h_deconv_concat, w1, keep_prob)
            conv1 = conv2d(deconv_concat, w1, tf.constant(1.0))
            h_conv = tf.nn.relu(conv1 + b1)
            # conv2 = conv2d(h_conv, w2, keep_prob)
            conv2 = conv2d(h_conv, w2, tf.constant(1.0))
            self.decoding_conv_layers[layer_index] = tf.nn.relu(conv2 + b2)
            input_x = self.decoding_conv_layers[layer_index]

            self.convs.append((conv1, conv2))

        ''' output map '''
        weight = weight_variable([1, 1, self.hps.init_feature_num, self.hps.class_num], stddev)
        bias = bias_variable([self.hps.class_num])
        conv = conv2d(input_x, weight, tf.constant(1.0))
        output_map = tf.nn.relu(conv + bias)
        self.decoding_conv_layers["out"] = output_map

        return output_map

    def _get_summary_op(self):
        # for i, (c1, c2) in enumerate(self.convs):
        #     tf.summary.image('summary_conv_%02d_01' % i, self._get_image_summary(c1))
        #     tf.summary.image('summary_conv_%02d_02' % i, self._get_image_summary(c2))

        # for k in self.pooling_layers.keys():
        #     tf.summary.image('summary_pool_%02d' % k, self._get_image_summary(self.pooling_layers[k]))

        # for k in self.deconv_layers.keys():
        #     tf.summary.image('summary_deconv_concat_%02d' % k, self._get_image_summary(self.deconv_layers[k]))

        # for k in self.encoding_conv_layers.keys():
        #     tf.summary.histogram("dw_convolution_%02d" % k + '/activations', self.encoding_conv_layers[k])

        # for k in self.decoding_conv_layers.keys():
        #     tf.summary.histogram("up_convolution_%s" % k + '/activations', self.decoding_conv_layers[k])

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('dice', self.dice)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.image('input', self._get_image_summary(self.x, 0))
        tf.summary.image('label', self._get_image_summary(self.y_one_hot, 1))
        tf.summary.image('output_prob', self._get_image_summary(self.prediction, 1))
        tf.summary.image('output', tf.cast(tf.expand_dims(tf.argmax(self.prediction, 3), axis=3), tf.float32))

        return tf.summary.merge_all()


    def _get_image_summary(self, img, idx=0):
        """
        Make an image summary for 4d tensor image with index idx
        """

        image_summary = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
        image_summary -= tf.reduce_min(image_summary)
        image_summary /= tf.reduce_max(image_summary)
        image_summary *= 255

        img_w = tf.shape(img)[1]
        img_h = tf.shape(img)[2]
        image_summary = tf.reshape(image_summary, tf.stack((img_w, img_h, 1)))
        image_summary = tf.transpose(image_summary, (2, 0, 1))
        image_summary = tf.reshape(image_summary, tf.stack((-1, img_w, img_h, 1)))
        return image_summary

    def _get_prediction(self):
        return tf.nn.softmax(self.logits + 1e-10)

    def _get_loss(self):
        if self.hps.loss_type == "cross_entropy":
            class_weights = tf.constant(np.array(self.hps.class_weights, dtype=np.float32))
            prob = tf.nn.softmax(logits + 1e-10)
            loss_map = tf.multiply(self.y_one_hot, tf.log(tf.clip_by_value(prob, 1e-10, 1.0)))
            loss = -tf.reduce_mean(tf.multiply(loss_map, class_weights))
        elif self.hps.loss_type == "dice_coefficient":
            eps = 1e-5
            prediction = pixel_wise_softmax_2(self.logits)
            intersection = tf.reduce_sum(prediction * self.y_one_hot)
            union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y_one_hot)
            loss = -(2 * intersection / (union))
        else:
            raise ValueError('Unknown loss function:' + self.hps.loss_type)
        return loss

    def _get_accuracy(self):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prediction, 3), tf.argmax(self.y_one_hot, 3)), tf.float32))

    def _get_dice(self):
        eps = 1e-5
        intersection = tf.reduce_sum(self.prediction * self.y_one_hot)
        union = eps + tf.reduce_sum(self.prediction) + tf.reduce_sum(self.y_one_hot)
        dice = 2 * intersection / union
        return dice

    def _get_optimizer(self):
        if self.hps.optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.hps.optimizer_type == 'momentum':
            self.learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                            global_step=tf.Variable(0),
                                                            decay_steps=self.hps.train_iters,
                                                            decay_rate=self.hps.decay_rate,
                                                            staircase=True)
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.hps.momentum)
        else:
            raise ValueError('Unknown optimizer:' + self.hps.optimizer_type)
        return optimizer

    def _get_train_op(self):
        optimizer = self._get_optimizer()
        train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        return train_op

    def restore(self, sess, save_path):
        ckpt = tf.train.get_checkpoint_state(self.hps.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver()
            saver.restore(sess, save_path)
            logging.info('model restore from: %s' % save_path)

    def save(self, sess, save_path):
        saver = tf.train.Saver()
        saver.save(sess, save_path)
        logging.info('model save at: %s' % save_path)


