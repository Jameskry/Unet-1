import tensorflow as tf
import numpy as np
import copy
import skimage.measure as measure
import logging
import nibabel as nib
import os
import medpy.metric as metric


class Operationer(object):
    def __init__(self, hps, model):
        self.hps = hps
        self.model = model

    def connected_filter(self, input_3D):
        output = copy.deepcopy(input_3D)
        label_3D = measure.label(input_3D > 0)
        max_label = np.max(label_3D)
        max_size = 0
        max_label_id = 0
        for label in range(1, max_label + 1):
            label_size = len(np.where(label_3D == label)[0])
            if label_size > max_size:
                max_size = label_size
                max_label_id = label
        output[np.where(label_3D != max_label_id)] = 0
        return output

    def save_data(self, data, ind):
        new_image = nib.Nifti1Image(data, np.ones((4, 4)))
        nib.save(new_image, os.path.join(self.hps.result_path, 'test-segmentation-' + str(ind) + '.nii'))

    def train(self, data_provider):
        logging.info('start training.............')
        save_path = os.path.join(self.hps.model_path, 'model.ckpt')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ''' restore from file '''
            if self.hps.need_restore:
                self.model.restore(sess, save_path)

            summary_writer = tf.summary.FileWriter(logdir=self.hps.log_path, graph=sess.graph)

            for epoch in range(self.hps.epochs):
                for step in range((epoch * self.hps.train_iters), ((epoch + 1) * (self.hps.train_iters))):
                    batch_x, batch_y = data_provider.next_train_batch()
                    _ = sess.run(self.model.train_op, feed_dict={self.model.x: batch_x, self.model.y: batch_y})
                    summary_info, loss, dice, acc = sess.run([self.model.summary_op, self.model.loss, self.model.dice,
                                                              self.model.accuracy], feed_dict={self.model.x: batch_x, self.model.y: batch_y})

                    if step % self.hps.report_step == 0:
                        logging.info('step {:}/{:}, loss={:.4f}, dice={:.4f}, acc={:.4f}'
                                     .format(step, self.hps.train_iters * self.hps.epochs, loss, dice, acc))
                    if step % 1000 == 0:
                        summary_writer.add_summary(summary_info, step)
                        summary_writer.flush()

                if self.hps.need_eval:
                    self.eval(sess, data_provider)
                self.model.save(sess, save_path)
            logging.info('end training.............')

            return save_path

    def eval(self, sess, data_provider):
        logging.info('start evaluating.............')
        img_count = 0
        dice_list = []
        for i in range(data_provider.test_data_len):
            cur_im_id = data_provider.test_data_list[i].split(' ')[0].split('/')[-1].split('.')[1]

            img_count += 1
            x_test = data_provider.get_test_image(i)
            output = sess.run(self.model.prediction, feed_dict={self.model.x: x_test})
            output = np.argmax(output, -1)
            output = np.reshape(output, (x_test.shape[1], x_test.shape[2]))
            if img_count == 1:
                output_3D = output[:, :, np.newaxis]
            else:
                output_new = output[:, :, np.newaxis]
                output_3D = np.concatenate((output_3D, output_new), axis=2)

                if i < data_provider.test_data_len - 1:
                    if data_provider.test_data_list[i + 1].split(' ')[0].split('/')[-1].split('.')[1] != cur_im_id:
                        output_3D = output_3D.astype(np.uint8)
                        output_3D = self.connected_filter(output_3D)
                        path = os.path.join(data_provider.liver_path, 'standard-segmentation-' + str(cur_im_id) + '.nii')
                        liver_region = nib.load(path).get_data()
                        dice = metric.dc(output_3D, liver_region)
                        dice_list.append(dice)
                        logging.info(cur_im_id + ' dice is ' + str(dice))
                        img_count = 0
                        continue
                if i == data_provider.test_data_len - 1:
                    output_3D = output_3D.astype(np.uint8)
                    output_3D = self.connected_filter(output_3D)
                    path = os.path.join(data_provider.liver_path, 'standard-segmentation-' + str(cur_im_id) + '.nii')
                    liver_region = nib.load(path).get_data()
                    dice = metric.dc(output_3D, liver_region)
                    dice_list.append(dice)
                    logging.info(cur_im_id + ' dice is ' + str(dice))
                    logging.info('dice per case is ' + str(np.mean(dice_list)))
                    break

    def test(self, data_provider):
        logging.info('start testing.............')
        save_path = os.path.join(self.hps.model_path, 'model.ckpt')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ''' restore from file '''
            self.model.restore(sess, save_path)

            img_count = 0
            dice_list = []
            for i in range(data_provider.test_data_len):
                cur_im_id = data_provider.test_data_list[i].split(' ')[0].split('/')[-1].split('.')[1]
                slice_id = data_provider.test_data_list[i].split(' ')[0].split('/')[-1].split('.')[2]

                img_count += 1
                x_test = data_provider.get_test_image(i)
                output = sess.run(self.model.prediction, feed_dict={self.model.x: x_test})
                output = np.argmax(output, -1)
                output = np.reshape(output, (x_test.shape[1], x_test.shape[2]))
                if img_count == 1:
                    output_3D = output[:, :, np.newaxis]
                else:
                    output_new = output[:, :, np.newaxis]
                    output_3D = np.concatenate((output_3D, output_new), axis=2)

                    if i < data_provider.test_data_len - 1:
                        if data_provider.test_data_list[i + 1].split(' ')[0].split('/')[-1].split('.')[1] != cur_im_id:
                            output_3D = output_3D.astype(np.uint8)
                            output_3D = self.connected_filter(output_3D)
                            self.save_data(output_3D, cur_im_id)
                        path = os.path.join(data_provider.liver_path,
                                            'standard-segmentation-' + str(cur_im_id) + '.nii')
                        liver_region = nib.load(path).get_data()
                        dice = metric.dc(output_3D, liver_region)
                        dice_list.append(dice)
                        logging.info(cur_im_id + ' dice is ' + str(dice))
                        img_count = 0
                        continue
                if i == data_provider.test_data_len - 1:
                    output_3D = output_3D.astype(np.uint8)
                    output_3D = self.connected_filter(output_3D)
                    self.save_data(output_3D, cur_im_id)
                    path = os.path.join(data_provider.liver_path, 'standard-segmentation-' + str(cur_im_id) + '.nii')
                    liver_region = nib.load(path).get_data()
                    dice = metric.dc(output_3D, liver_region)
                    dice_list.append(dice)
                    logging.info(cur_im_id + ' dice is ' + str(dice))
                    logging.info('dice per case is ' + str(np.mean(dice_list)))
                    break
