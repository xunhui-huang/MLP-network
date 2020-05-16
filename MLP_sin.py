import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import pylab
import math
import argparse

def paras():
    parse=argparse.ArgumentParser(description='Super parameters')
    parse.add_argument('-r',dest='RATIO',default=0.8,required=False)
    parse.add_argument('-l',dest='LEARNING_RATE',default=0.005,required=False)
    parse.add_argument('-n',dest='NUM_EPOCHS',default=100,required=False)
    parse.add_argument('-b',dest='BATCH_SIZE',default=20,required=False)
    args=parse.parse_args()
    return args

args=paras()
ratio=args.RATIO
lr=args.LEARNING_RATE
num_epochs=args.NUM_EPOCHS
num_batch=args.BATCH_SIZE

def get_data():
    input=[]
    label=[]
    for i in range(1000):
        x=random.uniform(0,2*np.pi)
        y=math.sin(x)
        input.append(x)
        label.append(y)
    return input,label

def inference(x):
    with tf.variable_scope('hidden1'):
        w=tf.get_variable('weights',shape=[1,16],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1),trainable=True)
        b=tf.get_variable('bias',shape=[16],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1),trainable=True)
        hidden1=tf.nn.relu(tf.matmul(x,w)+b)
    with tf.variable_scope('hidden2'):
        w=tf.get_variable('weights',shape=[16,16],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1),trainable=True)
        b=tf.get_variable('bias',shape=[16],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1),trainable=True)
        hidden2=tf.nn.relu(tf.matmul(hidden1,w)+b)  
    with tf.variable_scope('output'):
        w=tf.get_variable('weights',shape=[16,1],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1),trainable=True)
        b=tf.get_variable('bias',shape=[1],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1),trainable=True)
        output=tf.matmul(hidden2,w)+b
    return output

def train_and_test(ratio):
    input,label=get_data()
    num=len(label)
    offset=math.floor(ratio*num)
    train_x,train_y=input[0:offset],label[0:offset]
    test_x,test_y=input[offset:num],label[offset:num]
    return train_x,train_y,test_x,test_y

def get_batch(input,label):
    input_queue=tf.train.slice_input_producer([input,label],shuffle=True)
    x,y=tf.train.shuffle_batch(input_queue,batch_size=num_batch,capacity=100,min_after_dequeue=40)
    return x,y

train_x,train_y,test_x,test_y=train_and_test(ratio)
train_batch_x,train_batch_y=get_batch(train_x,train_y)
test_batch_x,test_batch_y=get_batch(test_x,test_y)

x=tf.placeholder(shape=[None,1],dtype=tf.float32,name='x')
y=tf.placeholder(shape=[None,1],dtype=tf.float32,name='y')

train_y_pred=inference(x)
train_loss=tf.reduce_mean(tf.square(train_y_pred-y))
tf.get_variable_scope().reuse_variables()
test_y_pred=inference(x)
test_loss=tf.reduce_mean(tf.square(test_y_pred-y))

train_op=tf.train.GradientDescentOptimizer(lr).minimize(train_loss)
init_op=tf.global_variables_initializer()
local_init_op=tf.local_variables_initializer()

gpu_options=tf.GPUOptions(allow_growth=True)
config=tf.ConfigProto(gpu_options=gpu_options)

with tf.Session(config=config) as sess:
    sess.run(init_op)
    sess.run(local_init_op)
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    epoch=0
    loss_for_plt=[]
    while epoch<num_epochs:
        epoch+=1
        train_input,train_label=sess.run([train_batch_x,train_batch_y])
        train_input=np.asarray(train_input)
        train_input=np.reshape(train_input,[num_batch,1])
        train_label=np.asarray(train_label)
        train_label=np.reshape(train_label,[num_batch,1])
        sess.run(train_op,feed_dict={x:train_input,y:train_label})
        if epoch%10==0:
            test_input,test_label=sess.run([test_batch_x,test_batch_y])
            test_input=np.asarray(test_input)
            test_label=np.asarray(test_label)
            test_input=np.reshape(test_input,[num_batch,1])
            test_label=np.reshape(test_label,[num_batch,1])
            loss=sess.run(test_loss,feed_dict={x:test_input,y:test_label})
            loss_for_plt.append(loss)
            print('epoch: %d,test loss is:%.5f' % (epoch,loss))
    epoch_for_plt=[i for i in range(0,num_epochs,10)]
    print('---Program ends')
    pylab.plot(epoch_for_plt,loss_for_plt,label='训练loss曲线')
    plt.axhline(linewidth=1,color='r')
    pylab.show()
    coord.request_stop()
    coord.join(threads)


        