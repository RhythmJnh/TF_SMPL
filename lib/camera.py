import tensorflow as tf
import numpy as np
import cv2

class Camera():
	def __init__(self, focal_length_x, focal_length_y, center_x, center_y, trans, axis_angle):
		self.fl_x = tf.constant(focal_length_x, dtype=tf.float32)
		self.fl_y = tf.constant(focal_length_y, dtype=tf.float32)
		self.cx = tf.constant(center_x, dtype=tf.float32)
		self.cy = tf.constant(center_y, dtype=tf.float32)
		self.trans = tf.constant(trans, dtype=tf.float32)
		self.rotm = tf.constant(cv2.Rodrigues(axis_angle)[0], dtype=tf.float32)

	def project(self, j3ds):
		j3ds = self.transform(j3ds)
		j3ds += 1e-8

		xs = tf.divide(j3ds[:, 0], j3ds[:, 2])
		ys = tf.divide(j3ds[:, 1], j3ds[:, 2])
		us = self.fl_x * xs + self.cx
		vs = self.fl_y * ys + self.cy
		res = tf.stack([us, vs], axis=1)

		return res

	def transform(self, j3ds):
		j3ds = tf.expand_dims(j3ds, axis=-1)
		rot = tf.tile(tf.reshape(self.rotm, [1, 3, 3]), [tf.shape(j3ds)[0], 1, 1])
		j3ds = tf.squeeze(tf.matmul(rot, j3ds))
		res = j3ds + tf.reshape(self.trans, [1, 3])

		return res
