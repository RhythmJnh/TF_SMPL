'''
  Author: Nianhong Jiao
'''

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import time
from lib.utils import *
from lib.camera import *
from lib.render_model import *
from lib.loadSMPL import loadSMPL
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('idx', type=int)
args = parser.parse_args()

def fit(img, j2ds, prior_path, model_dir, gen='n', camProject=None):
	smpl_joints_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 24]
	torso_ids = [2, 3, 8, 9]
	cids = range(12) + [13]
	flength = 5000
	center = np.array([img.shape[1]/2, img.shape[0]/2])

	sess = tf.Session()

	# Load model
	model = loadSMPL(model_dir, gender=gen)

	# Conf * base_weights(LSP data)
	base_weights = tf.reshape(tf.constant([1.,1.,0.5,0.5,1.,1.,1.,1.,1.,1.,1.,1.,1.], dtype=tf.float32), [-1, 1])

	# Load prior
	initial_param, pose_mean, pose_covariance = load_init_para(prior_path)
	pose_mean = tf.constant(pose_mean, dtype=tf.float32)
	pose_covariance = tf.constant(pose_covariance, dtype=tf.float32)

	param_shape = tf.Variable(initial_param[:10].reshape([1, -1]), dtype=tf.float32)
	param_rot = tf.Variable(initial_param[10:13].reshape([1, -1]), dtype=tf.float32)
	param_pose = tf.Variable(initial_param[13:82].reshape([1, -1]), dtype=tf.float32)
	param_trans = tf.Variable(initial_param[-3:].reshape([1, -1]), dtype=tf.float32)
	params = tf.concat([param_shape, param_rot, param_pose, param_trans], axis=1)

	# Get 3D joints
	j3ds, v = model.get_3d_joints(params, smpl_joints_ids)

	diff3d = tf.concat([[j3ds[9] - j3ds[3]], [j3ds[8] - j3ds[2]]], axis=0)
	m3h = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(diff3d), axis=1)))

	diff2d = tf.constant(np.array([j2ds[9] - j2ds[3], j2ds[8] - j2ds[2]]), dtype=tf.float32)
	m2h = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(diff2d), axis=1)))

	est_d = flength * (m3h / m2h)

	init_t = tf.concat([[0.], [0.], [est_d]], axis=0)
	# init_t = tf.reshape([[0.], [0.], [est_d]], [1, -1])
	rt = np.zeros(3)
	sess.run(tf.global_variables_initializer())
	init_t = init_t.eval(session=sess)

	camProject = Camera(flength, flength, center[0], center[1], init_t, rt)
	camProject.trans = tf.Variable(camProject.trans, dtype=tf.float32)
	sess.run(tf.global_variables_initializer())

	j2ds_model = tf.convert_to_tensor(camProject.project(j3ds))

	# VIS = True
	VIS = False
	def lc(j2ds_model):
		for i in range(0, 1):
			import copy
			tmp = copy.copy(img)
			for j2d in j2ds:
				x = int( j2d[1] )
				y = int( j2d[0] )

				if x > img.shape[0] or x > img.shape[1]:
					continue
				tmp[x:x+5, y:y+5, :] = np.array([0, 255, 0])

			for j2d in j2ds_model:
				x = int( j2d[1] )
				y = int( j2d[0] )

				if x > img.shape[0] or x > img.shape[1]:
					continue
				tmp[x:x+5, y:y+5, :] = np.array([255, 0, 0])
			plt.cla()
			plt.imshow(tmp)
			plt.pause(0.1)
	plt.show()

	if VIS:
		ls = lc
	else: ls = None

	objs = {}
	for idx, j in enumerate(torso_ids):
		objs['j2d_%d' % idx] = tf.reduce_sum(tf.square(j2ds_model[j] - j2ds[j]))
	objs['camt'] = tf.reduce_sum(tf.square(camProject.trans[2] - init_t[2]))
	loss = tf.reduce_mean(objs.values())
	optimizer = scipy_pt(loss=loss, var_list=[param_rot, camProject.trans], options={'ftol':0.001, 'maxiter':50, 'disp':False}, method='L-BFGS-B')
	optimizer.minimize(sess, fetches = [j2ds_model], loss_callback=ls)

	# # Rigid
	# objs = {}
	# for idx, j in enumerate(torso_ids):
	# 	objs['j2d_%d' % idx] = tf.reduce_sum(tf.square(j2ds_model[j] - j2ds[j]))
	# loss = tf.reduce_mean(objs.values())
	# optimizer = scipy_pt(loss=loss, var_list=[param_rot, param_trans], options={'ftol':0.001, 'maxiter':100, 'disp':False}, method='L-BFGS-B')
	# optimizer.minimize(sess, fetches = [j2ds_model], loss_callback=ls)

	# Non Rigid
	objs = {}
	pose_diff = tf.reshape(param_pose - pose_mean, [1, -1])
	objs['J2D_Loss'] = tf.reduce_sum(tf.square(j2ds_model - j2ds[cids]) * base_weights)
	objs['Prior_Loss'] = 5 * tf.squeeze(tf.matmul(tf.matmul(pose_diff, pose_covariance), tf.transpose(pose_diff))) 
	objs['Prior_Shape'] = 5 * tf.squeeze(tf.reduce_sum(tf.square(param_shape)))
	loss = tf.reduce_mean(objs.values())
	optimizer = scipy_pt(loss=loss, var_list=[param_rot, param_trans, param_pose, param_shape], options={'ftol':.001, 'maxiter':100, 'disp':False}, method='L-BFGS-B')
	optimizer.minimize(sess, fetches = [j2ds_model], loss_callback=ls)
	
	plt.close('all')

	verts = sess.run(v)
	faces = sess.run(model.f).astype(int)
	pose, betas, trans = sess.run([tf.concat([param_rot, param_pose], axis=1), param_shape, param_trans])
	verts = verts - trans
	model_params = {'trans': trans,
					'pose': pose,
					'shape': betas}	

	t = sess.run(camProject.trans)
	camRender = {'f': np.array([flength, flength]),
				 'c': center,
				 't': np.array(t),
				 'rt': rt}
	sess.close()
	del sess
	return model_params, verts, faces, camRender


def main():

	# Path
	img_dir     = './data/img'
	pose2d_dir  = './data/pose2d'
	# cam_path    = './data/camera/camera.mat'
	prior_path  = './data/prior/genericPrior.mat'
	model_dir   = './data/models'
	out_path    = './output'

	# Load camera
	# camRender, camProject = load_cam(cam_path)
	# print camRender.t, camRender.R_angles

	img_files = os.listdir(img_dir)
	idx = args.idx

	if idx is not None:
		i = img_files[idx]
		print (i)
		t0 = time()
		# Gender
		gen = 'f'
		# Load data term
		img_path = os.path.join(img_dir, i)
		pose2d_path = os.path.join(pose2d_dir, i.replace('.png', '.png_pose.npz'))
		img, j2ds = load_data(img_path, pose2d_path)

		model_params, verts, faces, camRender = fit(img, j2ds, prior_path, 
										 model_dir, gen=gen)#, camProject=camProject)
		t1 = time()
		print ('Run time: %.05fs' % (t1-t0))

		# Render
		h = img.shape[0]
		w = img.shape[1]
		# dist = np.abs((camRender.t[2]/1000.) - np.mean(verts, axis=0)[2])
		dist = np.abs((camRender['t'][2]) - np.mean(verts, axis=0)[2])
		res = (render_model(verts, faces, w, h, 
							camRender, far=20+dist, img=img) * 255.).astype('uint8')

		# Save
		out_img = os.path.join(out_path, 'img')
		cv2.imwrite(os.path.join(out_img, i), res)
		# cv2.imshow('test', res)
		# cv2.waitKey(0)
		save_to_obj(verts, faces, os.path.join(out_img.replace('img', 'obj'), i.replace('.png', '.obj')))
		save_to_pkl(model_params, os.path.join(out_img.replace('img', 'pkl'), i.replace('.png', '.pkl')))
		print ('Done..')
		# break


if __name__ == '__main__':
  main()
