
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import cv2
import input_data


# In[2]:


img_height=480
img_width=640
#Network Parameters
n_input=img_height*img_width*3
learning_rate=1e-4
training_iters=10000
batch_size=2
display_step=8
dropout=0.75
epoch=1

#tf graph input
x=tf.placeholder(tf.float32,[None,img_height,img_width,3])
y=tf.placeholder(tf.float32,[None,img_height,img_width,16])
keep_prob=tf.placeholder(tf.float32)


# In[3]:


'''def getTestPicArray(filename) :
    im = cv2.imread(filename)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_arr = np.array(im)
    nm = im_arr.reshape((1, n_input))
    nm = nm.astype(np.float32)
    return nm'''


# In[4]:


def conv2d(img,w,b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img,w,strides=[1,1,1,1],padding='SAME'),b))
def max_pool(img,k):
    return tf.nn.max_pool(img,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')
def conv2d_transpose(img,w,b,outputShape):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(img,w,output_shape=outputShape,strides=[1,1,1,1],padding='SAME'),b))


# In[5]:


wc1=tf.Variable(tf.random_normal([5,5,3,16]),name="wc1")
bc1=tf.Variable(tf.random_normal([16]),name="bc1")

wc2=tf.Variable(tf.random_normal([5,5,16,64]),name="wc2")
bc2=tf.Variable(tf.random_normal([64]),name="bc2")

wc3=tf.Variable(tf.random_normal([5,5,64,64]),name="wc3")
bc3=tf.Variable(tf.random_normal([64]),name="bc3")

wc4=tf.Variable(tf.random_normal([5,5,16,64]),name="wc4")
bc4=tf.Variable(tf.random_normal([16]),name="bc4")

wc5=tf.Variable(tf.random_normal([5,5,16,16]),name="wc5")
bc5=tf.Variable(tf.random_normal([16]),name="bc5")

#wout=tf.Variable(tf.random_normal([1,img_height,img_width,16]),name="wout")
#bout=tf.Variable(tf.random_normal([1,img_height,img_width,16]),name="bout")


# In[6]:


#Construct model
_X=tf.reshape(x,shape=[-1,img_height,img_width,3])
conv1_1=conv2d(_X,wc1,bc1)
conv1_2=max_pool(conv1_1,k=2)
conv1_2=tf.nn.dropout(conv1_2,keep_prob)

conv2_1=conv2d(conv1_2,wc2,bc2)
conv2_2=max_pool(conv2_1,k=2)
conv2_2=tf.nn.dropout(conv2_2,keep_prob)

conv3=conv2d(conv2_2,wc3,bc3)
conv3=max_pool(conv3,k=1)
conv3=tf.nn.dropout(conv3,keep_prob)

add1=tf.add(conv2_2,conv3)

conv_t1=tf.image.resize_bilinear(add1,conv2_1.get_shape().as_list()[1:3])
tmpShape=conv1_2.get_shape().as_list()
tmpShape[0]=batch_size
conv_t1=conv2d_transpose(conv_t1,wc4,bc4,outputShape=tmpShape)
conv_t1=tf.nn.dropout(conv_t1,keep_prob)

add2=tf.add(conv1_2,conv_t1)

conv_t2=tf.image.resize_bilinear(add2,conv1_1.get_shape().as_list()[1:3])
tmpShape=_X.get_shape().as_list()
tmpShape[0]=batch_size
tmpShape[3]=16
conv_t2=conv2d_transpose(conv_t2,wc5,bc5,outputShape=tmpShape)
conv_t2=tf.nn.dropout(conv_t2,keep_prob)

#predict=tf.add(tf.multiply(conv_t2,wout),bout)
cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=conv_t2,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# In[7]:


saver = tf.train.Saver({"wc1":wc1,"bc1":bc1,"wc2":wc2,"bc2":bc2,"wc3":wc3,"bc3":bc3,"wc4":wc4,"bc4":bc4,"wc5":wc5,"bc5":bc5})
init=tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# In[8]:


with tf.Session(config=config) as sess:
    sess.run(init)
    saver.restore(sess,"model/model.ckpt")
    for i in range(0,epoch):
        print("epoch:"+str(i)+"       !!!!!!!!!!!!")
        step=0
        now_at=0
        while step*batch_size<training_iters:
            X,Y=input_data.next_batch(
                    img_dir_path='mpii_human_pose_v1\\output_images\\',
                    index_path='train_data\\new_data.json',
                    img_height=480,img_width=640,
                    batch_size=batch_size)
            sess.run(optimizer,feed_dict = {x:X,y:Y,keep_prob:dropout})
            loss=sess.run(cost,feed_dict = {x:X,y:Y,keep_prob:1.})
            if step%display_step==0:
                print("Iter "+str(step*batch_size)+", Minibatch Loss="+"{:.6f}".format(loss))
            step+=1
            now_at+=batch_size
    save_path = saver.save(sess, "model/model_new.ckpt")
print("done!")

