import numpy as np
import datetime
import random
random.seed(datetime.datetime.now().second)


class DataProvider(object):
    def __init__(self, train_data_path, eval_data_path, test_data_path, batch_size):
        self.train_data_list = open(train_data_path, 'r').read().splitlines()
        self.eval_data_list = open(eval_data_path, 'r').read().splitlines()
        self.test_data_list = open(test_data_path, 'r').read().splitlines()
        self.train_data_len = len(self.train_data_list)
        self.eval_data_len = len(self.eval_data_list)
        self.test_data_len = len(self.test_data_list)
        self.liver_path = '/dfsdata/zhangyao_data/DB/LITS/volume/'
        self.batch_size = batch_size
        self.liver_size = 128

    def next_train_batch(self):
        image_list = []
        mask_list = []

        for i in range(self.batch_size):
            img, mask = self.get_random_image()

            ind = np.where(mask > 0)

            while len(ind[0]) == 0 or len(ind[0]) > 0 and ind[0].max() - ind[0].min() < self.liver_size or len(
                    ind[0]) > 0 and ind[1].max() - ind[1].min() < self.liver_size:
                img, mask = self.get_random_image()
                ind = np.where(mask > 0)

            image_list.append(img)
            mask_list.append(mask)

        image_batch = np.array(image_list, dtype=np.float32)
        mask_batch = np.array(mask_list, dtype=np.uint8)
        return image_batch, mask_batch

    def get_random_image(self):
        image_id = random.randint(0, self.train_data_len - 1)
        return self.get_image(image_id)

    def get_image(self, image_id):
        img = np.clip(np.load(self.train_data_list[image_id].split(' ')[0]), -75, 175)
        mask = np.load(self.train_data_list[image_id].split(' ')[1])
        mask = (mask > 0)
        return img[..., np.newaxis], mask

    def get_eval_image(self, image_id):
        img = np.clip(np.load(self.eval_data_list[image_id].split(' ')[0]), -75, 175)
        img = img[..., np.newaxis]
        img = img[np.newaxis,...]
        img = np.array(img, dtype=np.float32)
        return img

    def get_test_image(self, image_id):
        img = np.clip(np.load(self.test_data_list[image_id].split(' ')[0]), -75, 175)
        img = img[..., np.newaxis]
        img = img[np.newaxis,...]
        img = np.array(img, dtype=np.float32)
        return img
