'''
  Author: Nianhong Jiao
'''

import os
import cv2
import numpy as np
import scipy.io as sio
import pickle

from camera import Camera

def load_data(img_path, pose2d_path):
  img = cv2.imread(img_path)
  j2d = np.load(pose2d_path)
  j2d = j2d['pose'].T[:, :2]

  return img, j2d

def load_cam(cam_path):
  cam = sio.loadmat(cam_path, squeeze_me=True, struct_as_record=False)
  cam = cam['camera']

  camProject = Camera(cam.focal_length[0], cam.focal_length[1], 
    cam.principal_pt[0], cam.principal_pt[1], cam.t/1000., cam.R_angles)

  return cam, camProject 

def load_init_para(prior_path):
  pose_prior = sio.loadmat(prior_path, squeeze_me=True, struct_as_record=False)
  pose_mean = pose_prior['mean']
  pose_covariance = np.linalg.inv(pose_prior['covariance'])

  # Extra 3 for zero global
  zero_shape = np.ones([13]) * 1e-8
  zero_trans = np.ones([3]) * 1e-8
  initial_param = np.concatenate([zero_shape, pose_mean, zero_trans], axis=0)

  return initial_param, pose_mean, pose_covariance

def save_to_obj(verts, faces, path):
  # Frontal view
  verts[:, 1:3] = -verts[:, 1:3]
  with open(path, 'w') as fp:
    for v in verts:
      fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    for f in faces + 1:
      fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

def save_to_pkl(params, path):
  with open(path, 'w') as outf:
    pickle.dump(params, outf)