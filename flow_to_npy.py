#coding:UTF-8
import cv2
import numpy as np
import os
import pathlib
import glob
from multiprocessing import Pool,Lock
########
# ���ļ���Ҫ��hmdb-51Ŀ¼��ִ�У�ͬʱ��������Ŀ¼Ҳ������������
# Դ�ļ�����E:\\UCF-101-rand-croped��
########

#���ļ��м��ļ����ƶ����б��б�Ԫ�ص���ʽΪ{�ļ��������ļ����б�}
ROOT_DIR='ucf101'
SAVE_ROOT_DIR='ucf_flow_npy'
count=0
lock=Lock()
def read_dir_to_list(ROOT_DIR):
	file_list=os.listdir(ROOT_DIR)
	return file_list

def fun1(dir_name):
	global lock
	global count
	lock.acquire()
	try:
		count+=1
		count_tmp=count
		print('procesing %dth video'%count_tmp)
	finally:
		lock.release()
	if os.path.exists(SAVE_ROOT_DIR+os.sep+dir_name.split('_')[1]+os.sep+dir_name+'.npy'):
		print('Already exists,pass!')
		return
	flow_x_list=glob.glob(ROOT_DIR+os.sep+dir_name+os.sep+"flow_x_*.jpg")
	flow_y_list=glob.glob(ROOT_DIR+os.sep+dir_name+os.sep+"flow_y_*.jpg")
	if len(flow_x_list)<=0:
		return
	for i in range(len(flow_x_list)):
		#����ͼƬ����rgb��ʽ����ģ�����ͨ��������һģһ����������ֻ�����Ҷ�ͼ����
		x_path=flow_x_list[i]
		y_path=flow_y_list[i]
		#����ͼƬ
		x=cv2.imread(x_path,cv2.IMREAD_GRAYSCALE)
		y=cv2.imread(y_path,cv2.IMREAD_GRAYSCALE)
		flow_one_frame=np.dstack([x,y])
		shp=flow_one_frame.shape
		flow_one_frame=flow_one_frame.reshape(1,shp[0],shp[1],shp[2])
		if i==0:
			flow=flow_one_frame
		else:
			try:
				flow=np.concatenate((flow,flow_one_frame))
			except:
				print('jump!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
				with open('error.txt','a') as f:
					f.write(dir_name+'\n')
	#ֻ����ǰ80֡
	while(flow.shape[0]<80):
		flow=np.concatenate((flow,flow))
	shp=flow.shape
	flow=flow[0:80,0:shp[1],0:shp[2],0:shp[3]]
	shp=flow.shape
	flow=flow.reshape(1,shp[0],shp[1],shp[2],shp[3])
	#����������������dir_name��_�ֿ��ڶ�������Ϊ���ֵ��ļ�����
	sub_dir_name=dir_name.split('_')[1]
	sub_dir_path=SAVE_ROOT_DIR+os.sep+sub_dir_name

	if not os.path.exists(sub_dir_path):
		os.mkdir(sub_dir_path)
	#��ʽ������������������Ϊdir_name.npy
	np.save(pathlib.Path(sub_dir_path+os.sep+dir_name),flow)
	print('%dth video completed'%count_tmp)
'''
except:
	print('jump')
	with open('error.txt','a') as f:
		f.write(dir_name+'\n')
'''
#�½����̳�
if __name__=='__main__':
	global count
	file_list=read_dir_to_list(ROOT_DIR)
	pool=Pool(200)
	result=pool.map(fun1,file_list)
	pool.close()
	pool.join()

