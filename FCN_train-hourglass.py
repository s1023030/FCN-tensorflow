
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import cv2
import input_data


# In[2]:


img_height=480
img_width=360
#Network Parameters
n_input=img_height*img_width*3
learning_rate=1e-5
training_iters=10000
batch_size=4
display_step=5
dropout=0.9
epoch=20


#tf graph input
x=tf.placeholder(tf.float32,[None,img_height,img_width,3])
y=tf.placeholder(tf.float32,[None,img_height,img_width,16])
keep_prob=tf.placeholder(tf.float32)


# In[3]:


def conv2d(img,w,b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img,w,strides=[1,1,1,1],padding='SAME'),b))
def max_pool(img,k):
    return tf.nn.max_pool(img,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')
def conv2d_transpose(img,w,b,output_shape):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(img,w,output_shape=output_shape,strides=[1,1,1,1],padding='SAME'),b))
def cv_bottleneck(img,w1,w2):
    tmp_bn_1=tf.nn.relu(tf.nn.conv2d(img,w1,strides=[1,1,1,1],padding='SAME'))
    tmp_bn_2=tf.nn.conv2d(tmp_bn_1,w2,strides=[1,1,1,1],padding='SAME')
    return (img+tmp_bn_2)
def dc_bottleneck(img,w1,w2,output_shape):
    output_shape[3]=w1.get_shape().as_list()[2]
    tmp_bn_1=tf.nn.relu(tf.nn.conv2d_transpose(img,w1,output_shape=output_shape,strides=[1,1,1,1],padding='SAME'))
    output_shape[3]=w2.get_shape().as_list()[2]
    tmp_bn_2=tf.nn.conv2d_transpose(tmp_bn_1,w2,output_shape=output_shape,strides=[1,1,1,1],padding='SAME')
    return (img+tmp_bn_2)
def hourglass(img,wc1_1,wc1_2,bc1,wc2_1,wc2_2,bc2,wc3_1,wc3_2,bc3,wc4_1,wc4_2,bc4,wc5_1,wc5_2,bc5):
    tmp_cv_1=tf.nn.relu(tf.nn.bias_add(cv_bottleneck(img,wc1_1,wc1_2),bc1))
    tmp_cv_1_pooling=max_pool(tmp_cv_1,k=2)
    tmp_cv_1_pooling=tf.nn.dropout(tmp_cv_1_pooling,keep_prob)
    
    tmp_cv_2=tf.nn.relu(tf.nn.bias_add(cv_bottleneck(tmp_cv_1_pooling,wc2_1,wc2_2),bc2))
    tmp_cv_2_pooling=max_pool(tmp_cv_2,k=2)
    tmp_cv_2_pooling=tf.nn.dropout(tmp_cv_2_pooling,keep_prob)
    
    tmp_cv_3=tf.nn.relu(tf.nn.bias_add(cv_bottleneck(tmp_cv_2_pooling,wc3_1,wc3_2),bc3))
    tmp_cv_3=tf.nn.dropout(tmp_cv_3,keep_prob)
    
    tmp_rs_1=tf.image.resize_bilinear(tmp_cv_3,tmp_cv_2.get_shape().as_list()[1:3])
    tmpShape=tmp_cv_1_pooling.get_shape().as_list()
    tmpShape[0]=batch_size
    tmp_dc_1=tf.nn.relu(tf.nn.bias_add(dc_bottleneck(tmp_rs_1,wc4_1,wc4_2,output_shape=tmpShape),bc4))
    
    tmp_add_1=tf.add(tmp_dc_1,tmp_cv_1_pooling)
    tmp_add_1=tf.nn.dropout(tmp_add_1,keep_prob)
    
    tmp_rs_2=tf.image.resize_bilinear(tmp_add_1,tmp_cv_1.get_shape().as_list()[1:3])
    tmpShape=img.get_shape().as_list()
    tmpShape[0]=batch_size
    tmp_dc_2=tf.nn.relu(tf.nn.bias_add(dc_bottleneck(tmp_rs_2,wc5_1,wc5_2,output_shape=tmpShape),bc5))
    
    tmp_add_2=tf.add(tmp_dc_2,img)
    tmp_add_21=tf.nn.dropout(tmp_add_2,keep_prob)
    
    return (tmp_add_2)


# In[4]:


wc1=tf.Variable(tf.random_normal([7,7,3,32]),name="wc1")
bc1=tf.Variable(tf.random_normal([32]),name="bc1")

wh1_1_1=tf.Variable(tf.random_normal([5,5,32,32]),name="wh1_1_1")
wh1_1_2=tf.Variable(tf.random_normal([5,5,32,32]),name="wh1_1_2")

bh1_1=tf.Variable(tf.random_normal([32]),name="bh1_1")

wh1_2_1=tf.Variable(tf.random_normal([3,3,32,32]),name="wh1_2_1")
wh1_2_2=tf.Variable(tf.random_normal([3,3,32,32]),name="wh1_2_2")

bh1_2=tf.Variable(tf.random_normal([32]),name="bh1_2")

wh1_3_1=tf.Variable(tf.random_normal([3,3,32,32]),name="wh1_3_1")
wh1_3_2=tf.Variable(tf.random_normal([3,3,32,32]),name="wh1_3_2")

bh1_3=tf.Variable(tf.random_normal([32]),name="bh1_3")

wh1_4_1=tf.Variable(tf.random_normal([3,3,32,32]),name="wh1_4_1")
wh1_4_2=tf.Variable(tf.random_normal([3,3,32,32]),name="wh1_4_2")

bh1_4=tf.Variable(tf.random_normal([32]),name="bh1_4")

wh1_5_1=tf.Variable(tf.random_normal([3,3,32,32]),name="wh1_5_1")
wh1_5_2=tf.Variable(tf.random_normal([3,3,32,32]),name="wh1_5_2")

bh1_5=tf.Variable(tf.random_normal([32]),name="bh1_5")

wd1=tf.Variable(tf.random_normal([7,7,32,32]),name="wd1")
bd1=tf.Variable(tf.random_normal([32]),name="bd1")

wc2=tf.Variable(tf.random_normal([1,1,32,16]),name="wc2")
bc2=tf.Variable(tf.random_normal([16]),name="bc2")


# In[5]:


#Construct model
x=x/255.0
conv1=conv2d(x,wc1,bc1)
conv1_pooling=max_pool(conv1,k=7)
conv1_pooling=tf.nn.dropout(conv1_pooling,keep_prob)

hg_1=hourglass(conv1_pooling,wh1_1_1,wh1_1_2,bh1_1,wh1_2_1,wh1_2_2,bh1_2,wh1_3_1,wh1_3_2,bh1_3,wh1_4_1,wh1_4_2,bh1_4,wh1_5_1,wh1_5_2,bh1_5)

resize1=tf.image.resize_bilinear(hg_1,conv1.get_shape().as_list()[1:3])
tmpShape=x.get_shape().as_list()
tmpShape[0]=batch_size
tmpShape[3]=wd1.get_shape().as_list()[2]
dconv1=conv2d_transpose(resize1,wd1,bd1,output_shape=tmpShape)
dconv1=tf.nn.dropout(dconv1,keep_prob)

conv2=conv2d(dconv1,wc2,bc2)
conv2=tf.nn.dropout(conv2,keep_prob)

cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=conv2,labels=y))
optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)


# In[6]:


saver = tf.train.Saver({"wc1":wc1,"bc1":bc1,"wh1_1_1":wh1_1_1,"wh1_1_2":wh1_1_2,"bh1_1":bh1_1,"wh1_2_1":wh1_2_1,"wh1_2_2":wh1_2_2,
                        "bh1_2":bh1_2,"wh1_3_1":wh1_3_1,"wh1_3_2":wh1_3_2,"bh1_3":bh1_3,"wh1_4_1":wh1_4_1,"wh1_4_2":wh1_4_2,
                        "bh1_4":bh1_4,"wh1_5_1":wh1_5_1,"wh1_5_2":wh1_5_2,"bh1_5":bh1_5,
                        "wd1":wd1,"bd1":bd1,"wc2":wc2,"bc2":bc2
                       })
init=tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# In[ ]:


with tf.Session(config=config) as sess:
    sess.run(init)
    #saver.restore(sess,"model/model_hg_15.ckpt")
    #saver = tf.train.Saver()
    #saver.restore(sess,"model/model.ckpt")
    for i in range(0,epoch):
        print("epoch:"+str(i)+"       !!!!!!!!!!!!")
        step=0
        now_at=0
        while step*batch_size<training_iters:
            X=input_data.x_next_batch(
                    img_dir_path='mpii_human_pose_v1\\output_images360x480\\',
                    index_path='train_data\\new_data.json',
                    img_height=img_height,img_width=img_width,
                    batch_size=batch_size,now_at=now_at)
            Y=input_data.y_next_batch(
                    img_dir_path='mpii_human_pose_v1\\output_images360x480\\',
                    index_path='train_data\\new_data.json',
                    img_height=img_height,img_width=img_width,
                    batch_size=batch_size,now_at=now_at)
            sess.run(optimizer,feed_dict = {x:X,y:Y,keep_prob:dropout})
            loss=sess.run(cost,feed_dict = {x:X,y:Y,keep_prob:1.})
            if step%display_step==0:
                print("Iter "+str(step*batch_size)+", Minibatch Loss="+"{:.6f}".format(loss))
            step+=1
            now_at+=batch_size
        save_path = saver.save(sess, "model/model_hg_"+str(i)+".ckpt")
print("done!")

