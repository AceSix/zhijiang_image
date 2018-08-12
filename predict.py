import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# 加载模型

saver = tf.train.import_meta_graph("logs/model/model.ckpt-200.meta")
sess = tf.Session()
saver.restore(sess, "logs/model/model.ckpt-200")
image_size=200
graph=tf.get_default_graph()
input_x=graph.get_tensor_by_name("input:0")
keep_prob=graph.get_tensor_by_name("keep_prob:0")
score=graph.get_tensor_by_name("fc3/fc3:0")
softmax=graph.get_tensor_by_name("softmax:0")
output=graph.get_tensor_by_name("output:0")

def predict(image,sess):
    global softmax
    #image=cv2.imread(image)
    image = image[:, :, (2, 1, 0)]
    image=cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    image=np.reshape(image,[-1,image_size,image_size,3])
    out_score=sess.run(softmax,feed_dict={input_x:image,keep_prob:1})
    print(out_score)
    return out_score
#predict("data/wrong_92.jpeg",sess)
#predict("data/right_1.jpeg",sess)

from PIL import Image, ImageTk
import cv2
import time
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
t_start = time.time()
fps = 0
while cap.isOpened():
    ret, frame = cap.read()
    fps = fps + 1
    sfps = fps / ( time.time() - t_start )
    cv2.putText(frame, "FPS : " + str( int( sfps ) ), ( 10, 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 0 ), 2 )
    out_score=predict(frame,sess)
    cv2.putText(frame, "score: " + str(out_score), ( 200, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 0 ), 2 )
    cv2.imshow('frame',frame)
    cv2.waitKey(1000)