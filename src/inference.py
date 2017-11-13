import tensorflow as tf
from unet import HyperParams, Unet
from data_provider import DataProvider
from operationer import Operationer

train_data_path = '../data/train_slice_list.txt'
eval_data_path = '../data/eval_slice_list.txt'
test_data_path = '../data/test_slice_list.txt'
model_path = '../model/'
log_path = '../log/'
result_path = '../result/'

# height = 512
# width = 512
# channels = 1
# filter_size = 3
# layer_num = 5
# init_feature_num = 64
# class_num = 2
# class_weights = [1, 3]
# batch_size = 1
# pool_size = 2
# summary = True
# loss_type = 'cross_entropy'
# optimizer_type = 'adam'
# learning_rate = 0.0001
# decay_rate = 0.5
# momentum = 0.9
# train_iters = 10000
# epochs = 50

hps = HyperParams(model_path, 
                  log_path, 
                  result_path, 
                  channels=1, 
                  batch_size=1, 
                  filter_size=3, 
                  pool_size=2,
                  layer_num=5, 
                  init_feature_num=32, 
                  class_num=2, 
                  class_weights=[1, 3], 
                  loss_type='cross_entropy',
                  optimizer_type='adam', 
                  learning_rate=0.0001, 
                  decay_rate=0.5, 
                  momentum=0.9, 
                  train_iters=10000,
                  epochs=50, 
                  report_step=100, 
                  need_summary=True, 
                  need_restore=True, 
                  need_eval=False)

unet = Unet(hps=hps)
op = Operationer(hps=hps, model=unet)
data_provider = DataProvider(train_data_path, eval_data_path, test_data_path, hps.batch_size)

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', 'train or eval or test')
config = flags.FLAGS
if config.mode == 'train':
    op.train(data_provider)
elif config.mode == 'eval':
    op.eval(data_provider)
elif config.mode == 'test':
    op.test(data_provider)
else:
    raise ValueError('No such mode: ' + config.mode)

