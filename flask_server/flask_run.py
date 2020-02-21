from flask import Flask, request
from object_detection.utils import ops as utils_ops
import os
import numpy as np
import dlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
from gevent import monkey
monkey.patch_all()
import tensorflow as tf
app = Flask(__name__)
# F:/Coding_Tools/PyCharm 2018.2.4/flask_server/
PATH_TO_FROZEN_GRAPH = "frozen_inference_graph.pb"
PATH_TO_LABELS = "face_label_map.pbtxt"
IMAGE_SIZE = (256, 256)

detection_sess = tf.Session()

with detection_sess.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, IMAGE_SIZE[0], IMAGE_SIZE[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['im_urldetection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

###face_feature
face_feature_sess = tf.Session()
ff_pb_path = "face_recognition_model.pb"
with face_feature_sess.as_default():
    ff_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(ff_pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        ff_od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(ff_od_graph_def, name='')

        ff_images_placeholder = face_feature_sess.graph.get_tensor_by_name("input:0")
        ff_train_placeholder = face_feature_sess.graph.get_tensor_by_name("phase_train:0")

        ff_embeddings = face_feature_sess.graph.get_tensor_by_name("embeddings:0")

###face_landmark
face_landmark_sess = tf.Session()
fl_pb_path = "facial_landmark_SqueezeNet.pb"
with face_landmark_sess.as_default():
    fl_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(fl_pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        fl_od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(fl_od_graph_def, name='')

        landmark_tensor = face_feature_sess.graph.get_tensor_by_name("output/BiasAdd:0")

####################

# 加载人脸关键点检测模型
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()


@app.route("/face_landmark_tf", methods=['POST','GET'])
def face_landmark():
    ## 实现图片上传
    f = request.files.get('file')
    print(f)
    upload_path = os.path.join("tmp/tmp_landmark_tf." + f.filename.split(".")[-1])
    print(f.filename)
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)

    ## 实现人脸检测
    im_data = cv2.imread(upload_path)

    sp = im_data.shape ##

    im_data_re = cv2.resize(im_data, IMAGE_SIZE)
    output_dict = detection_sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(im_data_re, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    # print(output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'])
    x1, y1, x2, y2 = 0, 0, 0, 0
    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] > 0.5:
            bbox = output_dict['detection_boxes'][i]
            y1 = bbox[0]
            x1 = bbox[1]
            y2 = (bbox[2])
            x2 = (bbox[3])
            print(output_dict['detection_scores'][i], x1, y1, x2, y2)

            ## 提取人脸区域
            y1 = int(y1 * sp[0])
            print(sp[0], sp[1])
            x1 = int(x1 * sp[1])
            y2 = int(y2 * sp[0])
            x2 = int(x2 * sp[1])
            print(y1,y2,x1,x2)
            face_data = im_data[y1:y2, x1:x2]
            cv2.imwrite("tmp/face_landmark.jpg", face_data)

            im_data = cv2.cvtColor(face_data, cv2.COLOR_RGB2GRAY)
            im_data = cv2.resize(im_data, (64, 64))
            im_data = im_data.reshape(64, 64, 1)

            pred = face_landmark_sess.run(landmark_tensor, {"input_2:0":np.expand_dims(im_data, 0)})
            pred = pred[0]

            res=[]
            ##裁剪之后的人脸框中的坐标
            ##n
            for i in range(0, 136, 2):
                res.append(str((pred[i]  * (x2 - x1) + x1) / sp[1]))
                res.append(str((pred[i + 1]  * (y2 - y1) + y1) / sp[0]))

            res = ",".join(res)
            # for i in range(0, 136, 2):
            #     # if i == 48*2:
            #     cv2.circle(im_data, (int(pred[i] * 64), int(pred[i + 1] * 64)), 2, (0, 255, 0), 2)
            # cv2.imwrite("0_landmark.jpg", face_data)
            # return str(pred)
            return res
        return "error"

@app.route("/face_landmark", methods=['POST', 'GET'])
def face_landmark_dlib():
    ### 嘴巴、眼睛.casecade ###1）landmark 关键点定位、眼部关键点、粗位置、
    # 抠取、眼部关键点的回归。2）精细粒度眼部区域的回归。
    # 回归模型、人脸区域，---》提取眼部区域
    #
    ## http://------.jpg
    f = request.files.get("file")
    uplaod_path = os.path.join("tmp/tmp_landmark." + f.filename.split(".")[-1])
    f.save(uplaod_path)
    ##
    print(uplaod_path)

    ##RGB
    im_data = cv2.imread(uplaod_path)
    im_data = cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY)
    sp = im_data.shape

    rects = detector(im_data, 0)
    res = []
    for face in rects:
        shape = predictor(im_data, face)
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            # 坐标归一化
            ptx = (pt.x - x1) * 1.0 / (x2 - x1)
            pty = (pt.y - y1) * 1.0 / (y2 - y1)
            # 按人脸比例
            res.append(str(ptx))
            res.append(str(pty))
            # 按全图比例
            res.append(str(pt.x * 1.0 / sp[1]))
            res.append(str(pt.y * 1.0 / sp[0]))
        if res.__len__() == 136 * 2:
            res = ",".join(res)
            print(res)
            return res # 只检测第一个人脸
    return "error"





    return "error"

@app.route("/")
def helloworld():
    return '<h1>Hello World!</h1>'

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    f = request.files.get('file')
    print(f)
    upload_path = os.path.join("tmp/tmp." + f.filename.split(".")[-1])
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)
    return upload_path

@app.route("/face_detect")
def inference():

    im_url = request.args.get("url")

    im_data = cv2.imread(im_url)
    im_data = cv2.resize(im_data, IMAGE_SIZE)
    output_dict = detection_sess.run(tensor_dict, feed_dict={image_tensor:np.expand_dims(im_data, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    # print(output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'])
    x1, y1, x2, y2 = 0, 0, 0, 0
    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] > 0.5:
            bbox = output_dict['detection_boxes'][i]
            cate = output_dict['detection_classes'][i]
            y1 = bbox[0]
            x1 = bbox[1]
            y2 = (bbox[2])
            x2 = (bbox[3])
            # print(output_dict['detection_scores'][i], x1, y1, x2, y2)

    return str([x1, y1, x2, y2])

## 图像数据标准化
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def read_image(path):
    im_data = cv2.imread(path)
    im_data = prewhiten(im_data)
    im_data = cv2.resize(im_data, (160, 160))
    # 1 * h * w * 3
    return im_data

@app.route("/face_feature")
def __face_feature():
    im_data1 = read_image("D:/authentic_learningvideo/facenet-master/lfw_160\Abdel_Nasser_Assidi/Abdel_Nasser_Assidi_0002.png")
    im_data1 = np.expand_dims(im_data1, axis=0)

    emb1 = face_feature_sess.run(ff_embeddings, feed_dict={ff_images_placeholder:im_data1, ff_train_placeholder:False})

    strr = ", ".join(str(i) for i in emb1[0])

    return strr

@app.route("/face_dis")
def face_compare():
    im_data1 = read_image("D:/authentic_learningvideo/facenet-master/lfw_160/Abdel_Nasser_Assidi/Abdel_Nasser_Assidi_0002.png")
    im_data1 = np.expand_dims(im_data1, axis=0)
    emb1 = face_feature_sess.run(ff_embeddings, feed_dict={ff_images_placeholder:im_data1, ff_train_placeholder:False})

    im_data2 = read_image("D:/authentic_learningvideo/facenet-master/lfw_160/Aaron_Eckhart/Aaron_Eckhart_0001.png")
    im_data2 = np.expand_dims(im_data2, axis=0)
    emb2 = face_feature_sess.run(ff_embeddings, feed_dict={ff_images_placeholder:im_data2, ff_train_placeholder:False})

    dist = np.linalg.norm(emb1 - emb2)
    return str(dist)

@app.route('/face_register', methods=['POST', 'GET'])
def face_register():
    ## 实现图片上传
    f = request.files.get('file')
    print(f)
    upload_path = os.path.join("tmp/tmp." + f.filename.split(".")[-1])
    print(f.filename)
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)

    ## 实现人脸检测
    im_data = cv2.imread(upload_path)
    sp = im_data.shape
    im_data = cv2.resize(im_data, IMAGE_SIZE)
    output_dict = detection_sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(im_data, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    # print(output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'])
    x1, y1, x2, y2 = 0, 0, 0, 0
    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] > 0.5:
            bbox = output_dict['detection_boxes'][i]
            y1 = bbox[0]
            x1 = bbox[1]
            y2 = (bbox[2])
            x2 = (bbox[3])
            print(output_dict['detection_scores'][i], x1, y1, x2, y2)

            ## 提取人脸区域
            y1 = int(y1 * sp[0])
            print(sp[0], sp[1])
            x1 = int(x1 * sp[1])
            y2 = int(y2 * sp[0])
            x2 = int(x2 * sp[1])
            print(y1,y2,x1,x2)
            face_data = im_data[y1:y2, x1:x2]
            print("1")
            print(im_data.shape)
            print("qiehou",face_data.shape)
            im_data = prewhiten(im_data) # 预处理
            im_data = cv2.resize(im_data, (160, 160))
            im_data = np.expand_dims(im_data, axis=0)
            print("2")
            ## 人脸特征提取
            emb1 = face_feature_sess.run(ff_embeddings,
                                         feed_dict={ff_images_placeholder: im_data, ff_train_placeholder: False})
            strr = ",".join(str(i) for i in emb1[0])
            print("3")
            ## 写入txt
            with open("face/feature.txt", "w") as f:
                f.writelines(strr)
            f.close()
            mess = "success"
            break  # 只取一个人脸
        else:
            mess = "fail"

    return mess

@app.route('/face_login', methods=['POST', 'GET'])
def face_login():
    ## 图片上传
    ## 人脸检测
    ## 人脸特征提取
    ## 加载注册人脸（人脸签到，人脸数量很多，加载注册人脸放在face_Login，启动服务器加载/采用搜索引擎/ES）
    ## 同注册人脸相似性度量
    ## 返回度量结果

    f = request.files.get('file')
    print(f)
    upload_path = os.path.join("tmp/login_tmp." + f.filename.split(".")[-1])
    print(f.filename)
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)

    ## 实现人脸检测
    im_data = cv2.imread(upload_path)
    sp = im_data.shape
    im_data = cv2.resize(im_data, IMAGE_SIZE)
    output_dict = detection_sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(im_data, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    # print(output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'])
    x1, y1, x2, y2 = 0, 0, 0, 0
    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] > 0.5:
            bbox = output_dict['detection_boxes'][i]
            y1 = bbox[0]
            x1 = bbox[1]
            y2 = (bbox[2])
            x2 = (bbox[3])
            print(output_dict['detection_scores'][i], x1, y1, x2, y2)

            ## 提取人脸区域
            y1 = int(y1 * sp[0])
            print(sp[0], sp[1])
            x1 = int(x1 * sp[1])
            y2 = int(y2 * sp[0])
            x2 = int(x2 * sp[1])
            print(y1,y2,x1,x2)
            face_data = im_data[y1:y2, x1:x2]
            # print("1")
            # print(im_data.shape)
            # print("qiehou",face_data.shape)
            im_data = prewhiten(im_data) # 预处理
            im_data = cv2.resize(im_data, (160, 160))
            im_data = np.expand_dims(im_data, axis=0)
            # print("2")
            ## 人脸特征提取
            emb1 = face_feature_sess.run(ff_embeddings,
                                         feed_dict={ff_images_placeholder: im_data, ff_train_placeholder: False})

            with open("face/feature.txt","r") as f:
                fea_str = f.readlines()
                f.close()
            emb2_str = fea_str[0].split(",")
            emb2 = []
            for ss in emb2_str:
                emb2.append(float(ss))
            emb2 = np.array(emb2)

            dist = np.linalg.norm(emb1 - emb2)
            print("dist----------->",dist)
            if dist < 1.0:
                return "success"
            else:
                return "fail"
    return "fail"


if __name__ == '__main__':
    app.run(host='192.168.43.236', port=90, debug=True)
