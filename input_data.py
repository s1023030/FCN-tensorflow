import numpy as np
import cv2
import json
def draw_Gaussian_distribution(centerX,centerY,img_w,img_h,amplitude=255,stdx=100,stdy=100):
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
    final=amplitude*np.exp(combine)
    #print(final.shape)
    return final
def next_batch(img_dir_path='',index_path='',img_width=640,img_height=480,batch_size=10,now_at=0):
    X=np.empty([batch_size,img_height,img_width,3])
    Y=np.empty([batch_size,img_height,img_width,16])
    raw_datas= open(index_path).readlines()
    for i in range(0,batch_size):
        data=json.loads(raw_datas[now_at+i].strip())
        img_name=data['filename']
        img=cv2.imread(img_dir_path+img_name)
        X[i]=np.array(img)
        joints_pos=data['joint_pos']
        for j,joint in zip(range(0,len(joints_pos)),joints_pos):
            x=int(joints_pos[str(joint)][0])
            y=int(joints_pos[str(joint)][1])
            Y[i,:,:,j]draw_Gaussian_distribution(x,y,img_w=img_width,img_h=img_height))
    return X,Y
#xx,yy=next_batch(img_dir_path='mpii_human_pose_v1\\output_images\\',
#index_path='train_data\\new_data.json',batch_size=2,now_at=0)
#print(xx.shape)
