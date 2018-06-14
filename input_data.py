import numpy as np
import cv2
import json
def draw_Gaussian_distribution(centerX,centerY,img_w,img_h,amplitude=1,stdx=15,stdy=15):
    stdx=stdx^2
    stdy=stdy^2
    x,y = np.meshgrid(np.linspace(0,img_w,img_w), np.linspace(0,img_h,img_h))
    x=x-centerX
    y=y-centerY
    x=x*x
    y=y*y
    x=x/(2*stdx)
    y=y/(2*stdy)
    combine=(x+y)*(-1)
    final=amplitude*np.exp(combine)+1e-15
    return final
def xy_next_batch(img_dir_path='',index_path='',img_height=480,img_width=360,batch_size=10,now_at=0):
    X=np.empty([batch_size,img_height,img_width,3])
    Y=np.empty([batch_size,img_height,img_width,16])
    raw_datas= open(index_path).readlines()
    for i in range(0,batch_size):
        data=json.loads(raw_datas[now_at+i].strip())
        img_name=data['filename']
        img=cv2.imread(img_dir_path+img_name)
        X[i]=np.array(img)
        list_heatmaps=[]
        joints_pos=data['joint_pos']
        for j,joint in zip(range(0,len(joints_pos)),joints_pos):
            x=int(joints_pos[str(joint)][0])
            y=int(joints_pos[str(joint)][1])
            list_heatmaps.append(draw_Gaussian_distribution(x,y,img_w=img_width,img_h=img_height))
        numpy_heatmaps=np.array(cv2.merge(list_heatmaps[:]))
        Y[i]=numpy_heatmaps
    return X,Y

def x_next_batch(img_dir_path='',index_path='',img_height=480,img_width=360,batch_size=10,now_at=0):
    X=np.empty([batch_size,img_height,img_width,3])
    raw_datas= open(index_path).readlines()
    for i in range(0,batch_size):
        data=json.loads(raw_datas[now_at+i].strip())
        img_name=data['filename']
        img=cv2.imread(img_dir_path+img_name)
        X[i]=np.array(img)
    return X
def y_next_batch(img_dir_path='',index_path='',img_height=480,img_width=360,batch_size=10,now_at=0):
    Y=np.empty([batch_size,img_height,img_width,16])
    raw_datas= open(index_path).readlines()
    for i in range(0,batch_size):
        data=json.loads(raw_datas[now_at+i].strip())
        list_heatmaps=[]
        joints_pos=data['joint_pos']
        for j,joint in zip(range(0,len(joints_pos)),joints_pos):
            x=int(joints_pos[str(joint)][0])
            y=int(joints_pos[str(joint)][1])
            list_heatmaps.append(draw_Gaussian_distribution(x,y,img_w=img_width,img_h=img_height))
        numpy_heatmaps=np.array(cv2.merge(list_heatmaps[:]))
        Y[i]=numpy_heatmaps
    return Y