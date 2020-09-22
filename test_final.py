from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import time

from evaluations import get_val_data, perform_val
from models import ArcFaceModel
from utils import set_memory_growth, load_yaml, l2_norm


# flags.DEFINE_string('cfg_path', './configs/arc_res50.yaml', 'config file path')
# flags.DEFINE_string('gpu', '0', 'which gpu to use')
# flags.DEFINE_string('img_path', '', 'path to input image')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)
set_memory_growth()
cfg = load_yaml('./configs/arc_res50.yaml')

model = ArcFaceModel(size=cfg['input_size'],
                     backbone_type=cfg['backbone_type'],
                     training=False)

ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
if ckpt_path is not None:
    print("[*] load ckpt from {}".format(ckpt_path))
    model.load_weights(ckpt_path)
else:
    print("[*] Cannot find ckpt from {}.".format(ckpt_path))
    exit()
def get_embeds(img):
    start_time = time.time()
    # print("[*] Encode {} to ./output_embeds.npy".format(FLAGS.img_path))
    #img = cv2.imread(FLAGS.img_path)
    img = cv2.resize(img, (cfg['input_size'], cfg['input_size']))
    img = img.astype(np.float32) / 255.
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)
    embeds = l2_norm(model(img))
    #np.save('./output_embeds.npy', embeds)
    print("time taken: ",end=" ")
    print((time.time() - start_time))
    return embeds
