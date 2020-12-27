import tensorflow as tf
tf.enable_eager_execution()
import glob
import matplotlib.pyplot as plt

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.enable_eager_execution()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from skimage import io
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
import tqdm
import glob
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

tfds.disable_progress_bar()
AUTOTUNE = tf.data.experimental.AUTOTUNE


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

def preprocess_image_test(image):
    image_string = tf.read_file(image)
    image_decoded = tf.image.decode_jpeg(image_string, )
    image = normalize(image_decoded)
    return image



paths4 = walk_type('horse2zebra/horse2zebra/trainB/','*.jpg')#图片路径列表
filenames4 = tf.constant(paths4)
train_zebras = tf.data.Dataset.from_tensor_slices((filenames4))
train_zebras = train_zebras.map(preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(1000).batch(1)

paths2 = walk_type('horse2zebra/horse2zebra/trainA/','*.jpg')#图片路径列表
filenames2 = tf.constant(paths2)
train_horses = tf.data.Dataset.from_tensor_slices((filenames2))
train_horses = train_horses.map(preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(1000).batch(1)



OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)


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

#
# filepath = r'E:\xview2\cyclegan\horse2zebra\horse2zebra\testA'
# outpath = r'E:\xview2\cyclegan\horse2zebra\horse2zebra\testA_result'
# patchsize = 256
# for imagepath in os.listdir(filepath):
#     image = io.imread(os.path.join(filepath,imagepath))
#     semantic = generator_g(normalize(tf.expand_dims(image, 0)))[0]
#     io.imsave(os.path.join(outpath,imagepath), semantic)

EPOCHS = 5
def generate_images(model, test_input):
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # 获取范围在 [0, 1] 之间的像素值以绘制它。
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('./checkpoints/train')


@tf.function
def train_step(real_x, real_y):
    # persistent 设置为 Ture，因为 GradientTape 被多次应用于计算梯度。
    with tf.GradientTape(persistent=True) as tape:
        # 生成器 G 转换 X -> Y。
        # 生成器 F 转换 Y -> X。

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x 和 same_y 用于一致性损失。
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # 计算损失。
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
        # 总生成器损失 = 对抗性损失 + 循环损失。
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # 计算生成器和判别器损失。
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss,
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,
                                              discriminator_y.trainable_variables)

    # 将梯度应用于优化器。
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                              generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                              generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))
    return gen_g_loss,total_cycle_loss,identity_loss(real_y, same_y)

Gen_g_loss = []
Total_cycle_loss = []
Identity_loss = []
for epoch in range(EPOCHS):
  start = time.time()

  # n = 0
  for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
    b = train_step(image_x, image_y)
    Gen_g_loss.append(float(b[0]))
    Total_cycle_loss.append(float(b[1]))
    Identity_loss.append(float(b[2]))
    print(float(b[0]),float(b[1]),float(b[2]))
    # if n % 10 == 0:
    #   print ('.', end='')
    # n+=1
  clear_output(wait=True)

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))
import numpy as np
np.save('Gen_g_loss.npy',np.array(Gen_g_loss))
np.save('Total_cycle_loss.npy',np.array(Total_cycle_loss))
np.save('Identity_loss.npy',np.array(Identity_loss))
