import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
import random
import cv2
import glob
import image_slicer
import errno
from sklearn import svm
from sklearn import preprocessing
import sklearn as sk

test_data = []  
test_labels = []



train_num = 30

n_classes = 2
batch_size = 32

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 256 * 256 * 3])
    y = tf.placeholder(tf.float32, [None, n_classes])

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    
        # writer.add_session_log
             
   # with tf.Session() as sess: 
        
        weights = {'W_conv1':tf.Variable(tf.random_normal([10,10,3,32])),
               'W_conv2':tf.Variable(tf.random_normal([10,10,32,64])),
               'W_conv3':tf.Variable(tf.random_normal([10,10,64,128])),
               'W_conv4':tf.Variable(tf.random_normal([10,10,128,256])),
               'W_conv5':tf.Variable(tf.random_normal([10,10,256,512])),
               'W_fc1':tf.Variable(tf.random_normal(([8 * 8 * 512,4096]))),
               'W_fc2':tf.Variable(tf.random_normal(([4096,4096]))),
               'out':tf.Variable(tf.random_normal([4096, n_classes]))}
        
    
        biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_conv3':tf.Variable(tf.random_normal([128])),
               'b_conv4':tf.Variable(tf.random_normal([256])),
               'b_conv5':tf.Variable(tf.random_normal([512])),
               'b_fc1':tf.Variable(tf.random_normal([4096])),
               'b_fc2':tf.Variable(tf.random_normal([4096])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

        x = tf.reshape(x, shape=[-1, 256, 256, 3])
    
        conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
        print('conv1: ', conv1)
        conv1 = maxpool2d(conv1)
        print('pool1:', conv1)
        conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
        print('conv2: ', conv2)
        conv2 = maxpool2d(conv2)
        print('pool2:', conv2)
        conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
        print('conv3: ', conv3)
        conv3 = maxpool2d(conv3)
        print('pool3:', conv3)
        conv4 = tf.nn.relu(conv2d(conv3, weights['W_conv4']) + biases['b_conv4'])
        print('conv4: ', conv4)
        conv4 = maxpool2d(conv4)
        print('pool4:', conv4)  
        conv5 = tf.nn.relu(conv2d(conv4, weights['W_conv5']) + biases['b_conv5'])
        print('conv5: ', conv5)
        conv5 = maxpool2d(conv5)
        print('pool5:', conv5)          
        
        fc1 = tf.reshape(conv5,[-1, 8*8*512])
        fc1 = tf.nn.sigmoid(tf.matmul(fc1, weights['W_fc1'])+biases['b_fc1'])
        print('fc1:',fc1)
        fc1_keep = tf.nn.dropout(fc1, keep_rate)
        
        fc2 = tf.nn.sigmoid(tf.matmul(fc1_keep, weights['W_fc2'])+biases['b_fc2'])
        print('fc2:',fc2)
        fc2_keep = tf.nn.dropout(fc2, keep_rate)        
        
        output = tf.matmul(fc2_keep, weights['out'])+biases['out']
        
        
        #tf.summary.image('conv2', conv2_tensor)
       
                   
        
        return output
    


def improve_contrast(img):
    
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(4,4))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl,a,b))
    
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
      
    return final


def train_next_batch(num, time):
    #rate = random.uniform(0, 1)
    train_data = []
    train_labels = []
    gbm_total = batch_size/2 
    lgg_total = batch_size - gbm_total
    #index = random.sample(range(len(train_gbm)), num)
    index = random.sample(range(50),1)
    index = str(index)
    train_gbm = glob.glob('/Users/zhujunli/Desktop/data/ortraining-images/GBM_proj/'+index+'/*.jpg')    
    for i in range(gbm_total):
        training = cv2.imread(train_gbm[i]).astype('uint8')
        training = improve_contrast(training)
        #gray_image = cv2.cvtColor(training, cv2.COLOR_BGR2GRAY)
        if training.shape == (256, 256, 3): 
            training = training.astype(float)
            #training = cv2.cvtColor(training, cv2.COLOR_BGR2GRAY)
            train_data.append(training)
            train_labels.append([0,1])
            gbm_total = gbm_total - 1
        if gbm_total == 0:   
            break

    index = random.sample(range(70),1)
    index = str(index)
    train_lgg = glob.glob('/Users/zhujunli/Desktop/data/ortraining-images/LGG_proj/'+index+'/*.jpg')   
    for i in range(lgg_total):
        training = cv2.imread(train_lgg[i]).astype('uint8')
        training = improve_contrast(training)
       # gray_image = cv2.cvtColor(training, cv2.COLOR_BGR2GRAY)
        if training.shape == (256, 256, 3): 
            training = training.astype(float)
            train_data.append(training)
            train_labels.append([1,0])
            lgg_total = lgg_total - 1
        if lgg_total == 0:
            break
    train_total = len(train_data)
    train_data = np.asarray(train_data)  
    train_labels = np.asarray(train_labels)
    train_data = np.reshape(train_data, (num, 256 * 256 * 3))
    return train_data, train_labels
        
def test_next_batch(num, time):
    #rate = random.uniform(0, 1)
    test_data = []
    test_labels = [] 
    gbm_total = batch_size/2
    lgg_total = batch_size - gbm_total
    
    index = random.sample(range(40),1)
    index = str(index)
    test_gbm = glob.glob('/Users/zhujunli/Desktop/data/ortesting-images/GBM_small/'+index+'/*.jpg')       
    for i in range(gbm_total):
        testing = cv2.imread(test_gbm[i]).astype('uint8')
        testing = improve_contrast(testing)
        #gray_image = cv2.cvtColor(testing, cv2.COLOR_BGR2GRAY)
        if testing.shape == (256, 256, 3): 
            testing = testing.astype(float)
            test_data.append(testing)
            test_labels.append([0,1])
            gbm_total = gbm_total - 1
        if gbm_total == 0:
            break
          
    #index = random.sample(range(len(test_lgg)), num)
    index = random.sample(range(40),1)
    index = str(index)
    test_lgg = glob.glob('/Users/zhujunli/Desktop/data/ortesting-images/LGG_small/'+index+'/*.jpg')       
    for i in range(lgg_total):
        testing = cv2.imread(test_lgg[i]).astype('uint8')
        testing= improve_contrast(testing)
       # gray_image = cv2.cvtColor(testing, cv2.COLOR_BGR2GRAY)
        if testing.shape == (256, 256, 3): 
            testing = testing.astype(float)
            test_data.append(testing)
            test_labels.append([1,0])
            lgg_total = lgg_total - 1
        if lgg_total == 0:
            break
    
    test_data = np.asarray(test_data)  
    test_labels = np.asarray(test_labels)
    test_data = np.reshape(test_data, (num, 256 * 256 * 3))
    return test_data, test_labels

def train_neural_network_feature_extraction(x, layer):
    prediction = convolutional_neural_network(x)
    #print(prediction)
    cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits = prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 200
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter( "/Users/zhujunli/Desktop/data", sess.graph)
       
        sess.run(tf.initialize_all_variables())
        
        layer_tensor = sess.graph.get_tensor_by_name(layer)
        
        epoch_x = np.zeros(shape=(batch_size, 256 * 256 * 3))
        epoch_y = np.zeros(shape=(batch_size, 2))         
        for epoch in range(hm_epochs):
                       
            epoch_loss = 0
            
            for _ in range(int(train_num/batch_size)):            
                
                epoch_x, epoch_y = train_next_batch(batch_size,epoch)
                print(epoch_x.shape)
                
                _, c, layer_feature = sess.run([optimizer, cost, layer_tensor], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                
            #epoch_num = str(epoch)
            
            #print layer_feature.shape
            
        
            
            #percent = np.count_nonzero( layer_feature_norm)/float(32*32*32*128)
            
            #print percent
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            
        layer_feature = layer_feature.reshape(batch_size,4096)
        #layer_feature = layer_feature.reshape(32,16,16,4)  
        layer_feature_norm = layer_feature / layer_feature.max()
        #layer_feature_norm = layer_feature_norm.astype('uint8')
        #layer_tf = tf.convert_to_tensor(layer_feature_norm, np.uint8)
        #print layer_tf
        #result = sess.run(tf.summary.image('last',layer_tf))
        #layer_feature_norm = layer_feature_norm.reshape(32*1024)   
        
        np.savetxt("/Users/zhujunli/Desktop/data/features/feature_"+layer+".csv", layer_feature_norm, delimiter=",")
        np.savetxt("/Users/zhujunli/Desktop/data/features/label.csv", epoch_y, delimiter=",")
        saver.save(sess, '/Users/zhujunli/Desktop/data/features', global_step=hm_epochs,write_meta_graph=False)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
         
        test_data, test_labels = test_next_batch(batch_size,5)
        
        print('Accuracy:',accuracy.eval({x:test_data, y:test_labels}))        
        y_p = tf.argmax(prediction, 1)
        val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:test_data, y:test_labels})
        
        print "validation accuracy:", val_accuracy
        y_true = np.argmax(test_labels,1)
        print "Precision", sk.metrics.precision_score(y_true, y_pred)
        print "Recall", sk.metrics.recall_score(y_true, y_pred)
        print "f1_score", sk.metrics.f1_score(y_true, y_pred)        
train_neural_network_feature_extraction(x, 'Sigmoid_1:0')