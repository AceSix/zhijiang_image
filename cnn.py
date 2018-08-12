# -*- coding: utf-8 -*-
import tensorflow as tf

class CNN(object):
    def __init__(self,dropout_keep_prob=0.5):
        self.dropout_keep_prob = dropout_keep_prob

    def lenet_inference(self, x):
        conv1 = self.conv(x, 5, 5, 6, 1, 1, padding='SAME', name='conv1')
        pool1 = self.max_pool(conv1, 2, 2, 2, 2, padding='VALID', name='pool1')

        conv2 = self.conv(pool1, 5, 5, 16, 1, 1, name='conv2')
        pool2 = self.max_pool(conv2, 2, 2, 2, 2, padding='VALID', name ='pool2')

        flattened=tf.contrib.layers.flatten(pool2)
        fc1 = self.fc(flattened,flattened.get_shape()[-1], 120, name='fc1')
        fc1 = self.dropout(fc1, self.dropout_keep_prob)

        fc2 = self.fc(fc1, 120, 84, name='fc2')

        self.score = self.fc(fc2, 84, 30, relu=False, name='fc3')
        return self.score

    def loss(self,predict_y,batch_y):
        with tf.variable_scope("loss") as scope:
            print(predict_y.shape,batch_y.shape)
            self.loss = tf.sqrt(tf.reduce_sum(tf.square(predict_y-batch_y)))
            tf.summary.scalar(scope.name+'/loss', self.loss)
        return self.loss

    def evaluation(self,batch_y):
        with tf.variable_scope('accuracy') as scope:
            correct = tf.nn.in_top_k(predict_y, batch_y, 1)
            correct = tf.cast(correct, tf.float16)
            accuracy = tf.reduce_mean(correct)
            tf.summary.scalar(scope.name+'/accuracy', accuracy)
        return accuracy
    
    #训练
    def optimize(self, learning_rate):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
            train_op = optimizer.minimize(self.loss)
        return train_op
    


    """
    Helper methods
    """
    def conv(self,x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
        input_channels = int(x.get_shape()[-1])
        convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)
    
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels/groups, num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])
            
            # groups 多个GPU可以设置
            if groups == 1:
                conv = convolve(x, weights)
            else:
                input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
                weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
                conv = tf.concat(axis=3, values=output_groups)
    
            #bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
            bias = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(bias, name=scope.name)
            return relu
    
    def fc(self,x, num_in, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            print(scope.name)
            weights = tf.get_variable('weights', shape=[num_in, num_out])
            biases = tf.get_variable('biases', [num_out])
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
            if relu == True:
                relu = tf.nn.relu(act)
                return relu
            else:
                return act
    
    def max_pool(self,x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides = [1, stride_y, stride_x, 1],
                              padding = padding, name=name)
    
    def lrn(self,x, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)
    
    def dropout(self,x, keep_prob):
        return tf.nn.dropout(x, keep_prob)
    
    



