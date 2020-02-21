import tensorflow as tf
import cv2
import glob

import numpy as np

pb_path = "facial_landmark_SqueezeNet.pb"

sess = tf.Session()

with sess.as_default():
    with tf.gfile.FastGFile(pb_path, "rb") as f:
        grapg_def = sess.graph_def
        grapg_def.ParseFromString(f.read())
        tf.import_graph_def(grapg_def, name="")

im_list = glob.glob("tmp/64_te*")
landmark = sess.graph.get_tensor_by_name("output/BiasAdd:0")

for im_url in im_list:
    print(im_url)
    im_data = cv2.imread(im_url)
    im_data = cv2.cvtColor(im_data,cv2.COLOR_BGR2GRAY)
    im_data = im_data.reshape(64, 64, 1)
    # im_data = cv2.resize(im_data, (64, 64))

    pred = sess.run(landmark, {"input_2:0":np.expand_dims(im_data, 0)})

    print(pred)

    pred = pred[0]
    for i in range(0, 136, 2):
        # if i == 48*2:
        cv2.circle(im_data, (int(pred[i] * 64), int(pred[i + 1] * 64)), 2, (0, 255, 0), 2)

    # print(pred[44 * 2 + 1] * 128 - pred[46 * 2 + 1] * 128)

    cv2.imshow("11", im_data)
    cv2.waitKey(0)