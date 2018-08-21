import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_files(data_path):
    """
    构造image和label
    构造训练集和验证集
    """
    # attribute
    attribute_dict={}
    attribute_txt=os.path.join(data_path,"attributes_per_class.txt")
    with open(attribute_txt,'r') as f:
        for x in f.readlines():
            attribute_dict[x.strip().split()[0]]=[float(y) for y in x.strip().split()[1:]]

    train_txt=os.path.join(data_path,"train.txt")
    with open(train_txt,'r') as f:
        data_list=np.array([[x.strip().split()[0],x.strip().split()[1]] for x in f.readlines()])
    print(data_list.shape)

    val_image_list=[]
    val_label_list=[]
    train_image_list=[]
    train_label_list=[]

    label_set=np.array(list(set(data_list[:,1])))
    np.random.shuffle(label_set)
    val_label_set=label_set[:20]
    for data in data_list:
        image_path=os.path.join(data_path,"train",data[0])
        if data[1] in val_label_set:
            val_image_list.append(image_path)
            val_label_list.append(attribute_dict[data[1]])
            if len(attribute_dict[data[1]]) != 30:
                print("30")
        else:
            train_image_list.append(image_path)
            train_label_list.append(attribute_dict[data[1]])
            if len(attribute_dict[data[1]]) != 30:
                print(data[1])
                print(attribute_dict[data[1]])

    return np.array(train_image_list),np.array(train_label_list),np.array(val_image_list),np.array(val_label_list)

def get_batch(image,label,image_size,batch_size):
    
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.float32)
    #tf.cast()用来做类型转换

    input_queue = tf.train.slice_input_producer([image,label])
    # #加入队列

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])

    #jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档
    image = tf.image.decode_jpeg(image_contents,channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image,image_size,image_size)
    #image=tf.image.resize_images(image, (image_size, image_size), method=1)

    #对resize后的图片进行标准化处理
    #image = tf.image.per_image_standardization(image)

    image_batch,label_batch = tf.train.batch([image,label],batch_size = batch_size,num_threads=8,capacity = 3*batch_size)
    label_batch = tf.reshape(label_batch, [batch_size,30])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch,label_batch

if __name__=="__main__":
    data_path="./DatasetA"
    BATCH_SIZE = 4
    CAPACITY = 15
    IMAGE_SIZE=64

    train_image_list, train_label_list, val_image_list,val_label_list = get_files(data_path)
    print(train_image_list.shape,train_label_list.shape,val_image_list.shape,val_label_list.shape)
    image_batch,label_batch = get_batch(train_image_list,train_label_list,IMAGE_SIZE,BATCH_SIZE)

    with tf.Session() as sess:
        i=0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        try:
            while not coord.should_stop() and i<2:
            #提取出两个batch的图片并可视化。
                img,label = sess.run([image_batch,label_batch])

                for j in np.arange(BATCH_SIZE):
                    print('label:',label[j])
                    print(img[j,:,:,:].shape)
                    plt.imshow(img[j,:,:,:])
                    plt.show()
                i+=1
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)