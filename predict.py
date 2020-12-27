import tensorflow as tf
tf.enable_eager_execution()
from skimage import io
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

import glob

import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

# tfds.disable_progress_bar()
# AUTOTUNE = tf.data.experimental.AUTOTUNE


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize, x:x + windowSize])

def walk_type(path, file_type):
    paths = glob.glob(os.path.join(path,#存放图片的文件夹路径
                                   file_type # 文件类型
                                   )
                      )# path下所有file_type类型的文件的路径列表
    return paths


def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[256, 256, 3])
  return cropped_image

# 将图像归一化到区间 [-1, 1] 内。
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # 调整大小为 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # 随机裁剪到 256 x 256 x 3
  image = random_crop(image)

  # 随机镜像
  image = tf.image.random_flip_left_right(image)

  return image
def preprocess_image_train(image):
    image_string = tf.read_file(image
                                )
    image_decoded = tf.image.decode_jpeg(image_string, )
    image = random_jitter(image_decoded)
    image = normalize(image)
    return image
def preprocess_image_train1(image):
    image_string = tf.read_file(image
                                )
    image_decoded = tf.image.decode_png(image_string, )
    image = random_jitter(image_decoded)
    image = normalize(image)
    return image
def preprocess_image_test(image):
    image_string = tf.read_file(image)
    image_decoded = tf.image.decode_png(image_string, )
    image = normalize(image_decoded)
    return image


paths4 = walk_type(r'E:\xview2\cyclegan\NAS\Google_Image\google_harvey_256_0.2','*.png')#图片路径列表
filenames4 = tf.constant(paths4)
train_zebras = tf.data.Dataset.from_tensor_slices((filenames4))
train_zebras = train_zebras.map(preprocess_image_train1, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(1000).batch(1)

paths2 = walk_type(r'E:\xview2\cyclegan\256_post_harvey_test','*.png')#图片路径列表
filenames2 = tf.constant(paths2)
train_horses = tf.data.Dataset.from_tensor_slices((filenames2))
train_horses = train_horses.map(preprocess_image_train1, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(1000).batch(1)


sample_horse = next(iter(train_horses))
sample_zebra = next(iter(train_zebras))

plt.subplot(121)
plt.title('Horse')
plt.imshow(sample_horse[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Horse with random jitter')
plt.imshow(random_jitter(sample_horse[0]) * 0.5 + 0.5)
# plt.show()

plt.subplot(121)
plt.title('Zebra')
plt.imshow(sample_zebra[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Zebra with random jitter')
plt.imshow(random_jitter(sample_zebra[0]) * 0.5 + 0.5)

# plt.show()

OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

to_zebra = generator_g(sample_horse)
to_horse = generator_f(sample_zebra)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_horse, to_zebra, sample_zebra, to_horse]
title = ['Horse', 'To Zebra', 'Zebra', 'To Horse']

for i in range(len(imgs)):
  plt.subplot(2, 2, i+1)
  plt.title(title[i])
  if i % 2 == 0:
    plt.imshow(imgs[i][0] * 0.5 + 0.5)
  else:
    plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
# plt.show()

plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real zebra?')
plt.imshow(discriminator_y(sample_zebra)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real horse?')
plt.imshow(discriminator_x(sample_horse)[0, ..., -1], cmap='RdBu_r')

# plt.show()


LAMBDA = 10

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5
def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)
def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1
def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss


generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 如果存在检查点，恢复最新版本检查点
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')


filepath = r'E:\xview2\cyclegan\256_post_harvey_test'
outpath = r'C:\Users\84471\Desktop\result'
patchsize = 256
for imagepath in os.listdir(filepath):
    image = io.imread(os.path.join(filepath,imagepath))
    height = image.shape[0]
    pad_h = (int(height / patchsize) + 1) * patchsize
    width = image.shape[1]
    pad_w = (int(width / patchsize) + 1) * patchsize

    pad_img = np.zeros((pad_h, pad_w, 3),dtype='uint8')
    pad_img[:height, :width, :] = image
    tmp_pred = np.zeros((pad_h, pad_w,3),dtype='float32')
    for x, y, random_image in sliding_window(pad_img, patchsize, patchsize):
        semantic = generator_g(normalize(tf.expand_dims(random_image, 0)))[0]
        # visualize.display_instances(random_image, p['rois'], p['masks'], p['class_ids'], class_names, p['scores'])
        try:
            tmp_pred[y:y + patchsize, x:x + patchsize,:] = semantic
        except:
            tmp_pred[y:y + patchsize, x:x + patchsize,:] = np.zeros((patchsize, patchsize,3),dtype='float32')
    y_pred = tmp_pred[:height, :width,:]
    io.imsave(os.path.join(outpath,imagepath), y_pred)

