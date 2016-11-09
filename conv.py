import tensorflow as tf
import numpy as np
import csv
import timeit
#define shapes for the model
length = 1000
types = 37
out1 = 32
segement = 4
sh_w1 = [4,segement,1,out1]
out2 = 64
sh_w2 = [4,segement,out1,out2]
lastneuron = 1024
poolsize = 10
lastsize = length/(poolsize*poolsize)
savername = './37.ckpt'

def getdata(num,sess,feature,labels):
    ydata = list()
    examples = list()
    for j in range(num/100):
        e, label = sess.run([feature, labels])
        e = [map(float,list(e[i])) for i in range(100)]
        examples.append(e)
        ylab = np.zeros((100,types),dtype='float32')
        for i in range(100):
            ylab[i][int(label[i])] = 1.0
        ydata.append(ylab)
    xdata = np.reshape(examples,(num,4,length))
    ydata = np.reshape(ydata,(num,types))
    return xdata,ydata

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
    
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def maxpool(x):
    return tf.nn.max_pool(x,ksize=[1,1,poolsize,1],strides=[1,1,poolsize,1],padding='SAME')

'''
    import data
'''
directory = "./*.csv"
filename = "./result/result.csv"
csvfile = file(filename,'wb')
writer = csv.writer(csvfile)
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once(directory),
    shuffle=True)
line_reader = tf.TextLineReader(skip_header_lines=1)
_, csv_row = line_reader.read(filename_queue)

record_defaults = [[0],[""]]
min = 1000
capacity = min+3*100
labelnn,featuress = tf.decode_csv(csv_row, record_defaults=record_defaults)
labeln,features = tf.train.shuffle_batch([labelnn,featuress], batch_size = 100,min_after_dequeue = min, capacity = capacity)
'''
    construct the model
'''
# layer 1
y_      = tf.placeholder(tf.float32,shape = [None,types])
x       = tf.placeholder(tf.float32,shape = [None,4,length])
xd      = tf.reshape(x,[-1,4,length,1])
W_conv1 = weight_variable(sh_w1)
b_conv1 = bias_variable([out1])
h_conv1 = tf.nn.relu(conv2d(xd,W_conv1) + b_conv1)
h_pool1 = maxpool(h_conv1)
#layer 2
W_conv2 = weight_variable(sh_w2)
b_conv2 = bias_variable([out2])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = maxpool(h_conv2)
#connected layer 
W_fc1   = weight_variable([4*lastsize*out2,lastneuron])
b_fc1   = bias_variable([lastneuron])
h_pool2_flat= tf.reshape(h_pool2,[-1,4*lastsize*out2])
h_fc1   = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
#dropout
keep_prob   = tf.placeholder(tf.float32)
h_fc1_drop  = tf.nn.dropout(h_fc1,keep_prob)
#Readout layer
W_fc2   = weight_variable([lastneuron,types])
b_fc2   = bias_variable([types])
y       = tf.matmul(h_fc1_drop,W_fc2)+b_fc2
'''
    define training step 
'''
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
    start 
'''
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
for i in range(1,6001):
    #train
    xdata,ydata = getdata(100,sess,features,labeln)
    if(i%100==0):
        result = sess.run(accuracy, feed_dict={x: xdata,y_: ydata,keep_prob:1.0})
        print "step ",
        print i,
        print " accuracy = ",
        print result
        writer.writerow([i,result])
    sess.run(train_step,feed_dict={x:xdata,y_:ydata,keep_prob:0.5})
#Final test cases
xdata,ydata = getdata(1000,sess,features,labeln)
print "Final: ",
print(sess.run(accuracy, feed_dict={x: xdata,y_: ydata,keep_prob:1.0}))
saver = tf.train.Saver()
saver.save(sess,savername,global_step=1)
coord.request_stop()
coord.join(threads)
csvfile.close()
