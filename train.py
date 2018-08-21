import os
import numpy as np
import tensorflow as tf
import input_data
import net
import config as cfg
from timer import Timer
import datetime


attributes=[]
classes=[]
attribute_txt=os.path.join(cfg.DATA_PATH,"attributes_per_class.txt")
with open(attribute_txt,'r') as f:
    for x in f.readlines():
        classes.append(x.strip().split()[0])
        attributes.append([float(y) for y in x.strip().split()[1:]])
print(len(classes),len(attributes))

def train(data_path,logs_dir):
    timer = Timer()
    train_image_list,train_label_list,val_image_list,val_label_list = input_data.get_files(data_path)
    print("训练数据有%d,验证数据有%d" % (len(train_image_list),len(val_image_list)))
    train_image_batch,train_label_batch = input_data.get_batch(train_image_list,train_label_list,cfg.IMAGE_SIZE,cfg.BATCH_SIZE)
    test_image_batch,test_label_batch = input_data.get_batch(val_image_list,val_label_list,cfg.IMAGE_SIZE,cfg.BATCH_SIZE)
    x = tf.placeholder(tf.float32, shape=[None,cfg.IMAGE_SIZE,cfg.IMAGE_SIZE,3],name="input")
    y = tf.placeholder(tf.float32, shape=[None,30],name="output")
    keep_prob=tf.placeholder(tf.float32,name="keep_prob")
    
    model = net.DAPNet(attributes,classes)
    logits = model.inference(x)
    loss = model.loss(y,logits)
    acc = model.accuary(y,logits)
    train_op = model.optimize(cfg.LEARNING_RATE)
    
    summary_op = tf.summary.merge_all()
    
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(logs_dir+'/train/',sess.graph)
        test_writer = tf.summary.FileWriter(logs_dir+'/test/',sess.graph)

        saver=tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        
        # start queue runner
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        try:
            for step in np.arange(cfg.MAX_ITER):
                if coord.should_stop():
                    break
                tra_images,tra_labels = sess.run([train_image_batch, train_label_batch])
                val_images, val_labels = sess.run([test_image_batch, test_label_batch])
                sess.run(train_op,feed_dict={x:tra_images, y:tra_labels,keep_prob:0.5})
                if step % 50==0:
                    timer.tic()
                    tra_acc,tra_loss,summary_str = sess.run([acc,loss,summary_op],feed_dict={x:tra_images, y:tra_labels,keep_prob:1.0})
                    train_writer.add_summary(summary_str, step)

                    val_acc,val_loss,summary_str = sess.run([acc,loss,summary_op],feed_dict={x:val_images, y:val_labels,keep_prob:1.0})
                    test_writer.add_summary(summary_str, step)
                    timer.toc()
                    log_str = '''{}, Step={},train_loss={:5.3f},train_acc={:5.3f},val_loss={:5.3f},val_acc={:5.3f}
                    Speed: {:.3f}s/iter, Remain: {}'''.format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        int(step),
                        tra_loss,
                        tra_acc,
                        val_loss,
                        val_acc,
                        timer.average_time,
                        timer.remain(step, cfg.MAX_ITER))
                    print(log_str)

            model_path=os.path.join(logs_dir+'model/', 'model.ckpt')
            saver.save(sess,model_path,global_step=cfg.MAX_ITER)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop() 
        # stop queue runner
        coord.request_stop()
        coord.join(threads)
        
if __name__=="__main__":
    data_path="./DatasetA_train_20180813"
    logs_dir="./logs"
    train(data_path,logs_dir)
