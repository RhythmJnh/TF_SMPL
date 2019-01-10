'''
  Author: Nianhong Jiao
'''

import os
import numpy as np
import pickle as pkl
import tensorflow as tf

class loadSMPL():
    '''
        For details comments see SMPL_MODEL npModel.py
    '''
    def __init__(self, model_dir, gender='n'):
        data = pkl.load(open(os.path.join(model_dir, 'basicModel_%s.pkl' % gender)))
        self.v_template = tf.constant(data['v_template'], dtype=tf.float32)
        self.f = tf.constant(data['f'], dtype=tf.float32)
        self.shapedirs = tf.constant(data['shapedirs'].r, dtype=tf.float32)
        self.posedirs = tf.constant(data['posedirs'], dtype=tf.float32)
        self.J_regressor = tf.constant(data['J_regressor'].todense(), dtype=tf.float32)
        self.parent_ids = data['kintree_table'][0].astype(int)
        self.weights = tf.constant(data['weights'], dtype=tf.float32)

    def get_3d_joints(self, params, index_list):
        betas = params[:, :10]
        pose = params[:, 10:82]
        trans = params[:, -3:]

        g, v = self.tfGenerate(betas, pose, trans)
        g = tf.convert_to_tensor(g)
        res = g[:, :3, 3] + trans

        # Index of head, 411
        res = tf.concat([res, tf.reshape(v[411, :], [1, -1])], axis=0)
        tmp = []
        for i in index_list:
          tmp.append(res[i, :])
        res = tf.convert_to_tensor(tmp)

        return res, v + trans

    def tfGenerate(self, betas, pose, trans):
        betas = tf.cast(betas, dtype=tf.float32)
        pose = tf.cast(pose, dtype=tf.float32)
        trans = tf.cast(trans, dtype=tf.float32)
        v_shaped = tf.expand_dims(self.v_template, axis=0) + tf.einsum('ijk,lk->lij', self.shapedirs, betas)

        I_pose = tf.zeros([23, 3], dtype=tf.float32)
        I = rodrigues(tf.reshape(I_pose, (-1, 1, 3)))
        I_matrix = tf.reshape(I, [1, -1])

        R = rodrigues(tf.reshape(pose, [-1, 1, 3]))
        R_matrix = tf.reshape(R[1:], [1, -1])

        rot_matrix = R_matrix - I_matrix

        v_posed = v_shaped + tf.einsum('ijk,lk->lij', self.posedirs, rot_matrix)

        J = tf.einsum('ij,ljk->lik', self.J_regressor, v_shaped)

        G = [0] * 24
        G[0] = with_zeros(tf.concat([R[0], tf.reshape(J[:, 0], [3, 1])], axis=1))
        for idx, pid in enumerate(self.parent_ids[1:]):
          G[idx+1] = tf.matmul(G[pid], with_zeros(tf.concat([R[idx+1, :3, :3], tf.reshape(J[:, idx+1]-J[:, pid], [3, 1])], axis=1)))

        Gs = tf.stack(G, axis=0)
        Gt = Gs - pack(tf.matmul(Gs, tf.reshape(tf.concat((J[0], tf.zeros((24, 1), dtype=tf.float32)), axis=1), [24, 4, 1])))

        T = tf.tensordot(self.weights, Gt, axes=((1), (0)))
        rest_shape = tf.concat((v_posed[0], tf.ones((v_posed.get_shape().as_list()[1], 1), dtype=tf.float32)), axis=1)
        v = tf.squeeze(tf.matmul(T, tf.reshape(rest_shape, (-1, 4, 1))))
        verts = v[:, :3] + trans

        return G, verts

def rodrigues(r):
    '''
        Rodrigues in batch mode.
        Input: Nx1x3 rotation vectors
        Output: Nx3x3 rotation matrix
    '''
    theta = tf.norm(r + tf.random_normal(r.shape, 0, 1e-8, dtype=tf.float32), axis=(1, 2), keepdims=True)
    r_hat = r / theta
    cos = tf.cos(theta)
    z_stick = tf.zeros(theta.get_shape().as_list()[0], dtype=tf.float32)
    m = tf.stack(
        (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
        -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), axis=1)
    m = tf.reshape(m, (-1, 3, 3))
    i_cube = tf.expand_dims(tf.eye(3, dtype=tf.float32), axis=0) + tf.zeros(
        (theta.get_shape().as_list()[0], 3, 3), dtype=tf.float32)
    A = tf.transpose(r_hat, (0, 2, 1))
    B = r_hat
    dot = tf.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + tf.sin(theta) * m

    return R

def with_zeros(x):
  xs = tf.concat((x, tf.constant([[0., 0., 0., 1.]], dtype=tf.float32)), axis=0)
  return xs

def pack(x):
  xs = tf.concat((tf.zeros((x.get_shape().as_list()[0], 4, 3), dtype=tf.float32), x), axis=2)
  return xs