import numpy as np 
import tensorflow as tf
import config as cfg 

slim=tf.contrib.slim

class DAPNet(object):
    """
    Direct Attributes Prediction
    """

    def __init__(self,attributes,classes):
        self.attributes=attributes
        self.classes = classes

    def inference(self,x,keep_prob=0.5,is_training=True):
        """
        DAP网络结构
        输出属性空间
        """
        with slim.arg_scope([slim.conv2d],padding="SAME",
        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
        weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.conv2d(x,32,[5,5],scope="conv_1")
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool_2')
            net = slim.conv2d(net, 64, [5, 5], scope='conv_3')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool_4')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1024, scope='fc_5')
            net = slim.dropout(net, keep_prob, is_training=is_training,
                       scope='dropout_6')
            score = slim.fully_connected(net, len(self.attributes[0]), activation_fn=None,
                                  scope='fc_7')
            return score

    def accuary(self,batch_y,score):
        """
        计算预测准确率
        最近score和attributes之间的最近距离作为预测值
        
        """
        # self.attributes(shape=(num_classes,att_length))
        # score shape=(batch_size,att_length)
        score_index=tf.argmax(tf.matmul(self.attributes,tf.transpose(score)),axis=0) #batch_size
        batch_y_index=tf.argmax(tf.matmul(self.attributes,tf.transpose(batch_y)),axis=0)
        print(score_index,batch_y_index)
        correct = tf.equal(score_index, batch_y_index)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        return accuracy

    def loss(self,batch_y,score):
        """
        计算loss
        score和lable之间的距离作为loss
        """
        print(batch_y,score)
        self.loss = (1.0/(1.0+cfg.REGUL)) * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_y, logits=score))
        return self.loss
    
    def optimize(self,learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        train_op = optimizer.minimize(self.loss)
        return train_op
