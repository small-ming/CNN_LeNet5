import tensorflow as tf
import cv2
import numpy as np

# img = cv2.imread("IMG_3250.JPG")
img = cv2.imread("G:\OpenCV+TensorFlow\code\lena.jpg")
img = np.array(img,dtype=np.float32)
image=tf.image.resize_images(img,(512,512),2)
x_image=tf.reshape(image,[1,512,512,3])
# x_image=tf.resize(image,[1,512,512,3])
# x_image=tf.reshape(img,[1,-1,-1,3])
print(img.shape)
print(x_image.shape)
# '''
# x_image=tf.reshape(img,[1,512,512,3])
# x_image=tf.reshape(img,[1,256,256,3])

filter = tf.Variable(tf.ones([7, 7, 3, 1]))

init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)
    res = tf.nn.conv2d(x_image, filter, strides=[1, 2, 2, 1], padding='SAME')
    res = tf.nn.max_pool(res, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    res_image = sess.run(tf.reshape(res,[128,128]))/128 + 1
    # res_image = sess.run(tf.reshape(res,[64,64]))/64 + 1

cv2.imshow("lena",res_image.astype('uint8'))
cv2.waitKey()
# '''