import tensorflow as tf
import os
import numpy as np
import config as cfg
import cv2
from timer import Timer
import datetime


# 加载模型

saver = tf.train.import_meta_graph("logs/model/model.ckpt-200.meta")
sess = tf.Session()
saver.restore(sess, "logs/model/model.ckpt-200")

graph=tf.get_default_graph()
input_x=graph.get_tensor_by_name("input:0")
keep_prob=graph.get_tensor_by_name("keep_prob:0")
output=graph.get_tensor_by_name("score/BiasAdd:0")

def predict(image,sess):
    global output
    image=cv2.imread(image)
    image = image[:, :, (2, 1, 0)]
    image=cv2.resize(image, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    image=np.reshape(image,[-1,cfg.IMAGE_SIZE,cfg.IMAGE_SIZE,3])

    score=sess.run(output,feed_dict={input_x:image,keep_prob:1})
    return score
timer=Timer()
test_txt=os.path.join(cfg.TEST_PATH,"DatasetA_test","image.txt")
submit_txt=os.path.join(cfg.TEST_PATH,"DatasetA_test","submit.txt")
with open(test_txt,'r') as f:
    images=f.readlines()
    #print("测试集数量：",len(f.readlines()))
    for i in range(len(images)):
        timer.tic()
        x=images[i]
        num=len(images)
        image_path=os.path.join(cfg.TEST_PATH,"DatasetA_test","test",x.strip())
        score=predict(image_path,sess)
        score_index=np.argmax(np.matmul(cfg.ATTRIBUTES,score.T),axis=0) #batch_size
        y=cfg.CLASSES[score_index[0]]
        print(y)
        with open(submit_txt,'a') as f:
            f.write(x.strip()+"\t"+y+"\n")
        timer.toc()
        log_str = '''{}, num={},Speed: {:.3f}s/image, Remain: {}'''.format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        i,
                        timer.average_time,
                        timer.remain(i, num))
        print(log_str)
        break
        