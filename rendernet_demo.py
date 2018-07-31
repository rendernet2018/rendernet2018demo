import os
import argparse
import math
import numpy as np
from scipy import misc

import tensorflow as tf

import binvox_rw
# =======================================================================================================
# RenderNet Phong shading demo
# The network output a normal map from a 3D voxel.
# The normal map is then used to create phong shading with lighting control.
# ======================================================================================================

# Phong shading paramters
AMBIENT_IN = (0.1)
K_DIFFUSE = .9
LIGHT_COL = np.array([[1., 1., 1.]])


def load_graph(frozen_graph_filename):
  with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
      tf.import_graph_def(graph_def, name="")
  return graph


def compute_pose_param(azimuth, elevation, radius):
  phi = azimuth * math.pi / 180.0
  theta = (90 - elevation) * math.pi / 180
  param = np.array([phi, theta, 3.3 / radius])
  param = np.expand_dims(param, axis=0)
  return param


def render(azimuth, elevation, radius, sess, voxel, light_dir,
           render_dir, count, light_azimuth, light_elevation,
           model_name):
  param = compute_pose_param(azimuth, elevation, radius)

  # Creating normal map
  rendered_samples = sess.run("encoder/output:0",
                              feed_dict={"real_model_in:0": voxel,
                                         "view_name:0": param,
                                         "patch_size:0": 128,
                                         "is_training:0": False})
  # Create phong shaded images
  img_phong = np_phong_composite(rendered_samples * 255.,
                                 light_dir, LIGHT_COL,
                                 AMBIENT_IN, K_DIFFUSE)

  image_out = np.clip(255. * img_phong[0], 0, 255).astype(np.uint8)

  save_path = os.path.join(render_dir,
                           str(count).zfill(3) + "_" + model_name +
                           "_pose_%f_%f_%f_light_%f_%f.png" %
                           (azimuth, elevation, radius,
                            light_azimuth, light_elevation))
  print(save_path)
  misc.imsave(save_path, image_out)


def np_mask(image_in):
  mask = np.linalg.norm(image_in, axis=3, keepdims=True)
  sigmoid = lambda x: 1. / (1. + np.exp(-x))
  mask = sigmoid(mask - 185)
  return mask


def np_mask_white(image_in):
  mask = np.linalg.norm(255. - image_in, axis=3, keepdims=True)
  sigmoid = lambda x: 1. / (1. + np.exp(-x))
  mask = sigmoid(mask - 80)
  return mask


def np_phong_shading(img_batch, light_dir, light_col, k_diffuse):
  # Phong shading implementation:

  # Convert normal map image to normals, and reshape into N-by-3 matrix.
  normals_ish = img_batch / 255. - 0.5
  normals_ish_vec = normals_ish.reshape([-1, 3])

  # Normalise each row => unit normals in each row
  normals_vec = (normals_ish_vec /
                 np.linalg.norm(normals_ish_vec, axis=1)[:, np.newaxis])
  # print("normals_vec.shape = " + str(normals_vec.shape))

  # Normalise light directions.
  light_dir /= np.linalg.norm(light_dir, axis=1).reshape([-1, 1])

  # Repeat each light for all pixels in an image.
  light_dir = np.repeat(light_dir, np.prod(img_batch.shape[1:3]), 0)
  light_col = np.repeat(light_col, np.prod(img_batch.shape[1:3]), 0)
  print("light_dir.shape = " + str(light_dir.shape))
  print("light_col.shape = " + str(light_col.shape))

  # Diffuse shading is basically N.L ...
  diffuse_vec = np.sum(np.multiply(normals_vec, light_dir),
                       axis=1, keepdims=True)
  print("diffuse_vec.shape = " + str(diffuse_vec.shape))

  # ... clamping negative values ...
  diffuse_vec = np.maximum(diffuse_vec, 0.0)

  # ... repeated for each colour channel ...
  diffuse_vec = np.repeat(diffuse_vec, 3, 1)
  print("diffuse_vec.shape = " + str(diffuse_vec.shape))

  # ... multiplied by the light colour and diffuse reflectivity (k_diffuse) ...
  diffuse_col_vec = k_diffuse * np.multiply(diffuse_vec, light_col)
  print("diffuse_col_vec.shape = " + str(diffuse_col_vec.shape))

  # ... and reshaped back into the batch shape.
  diffuse = np.reshape(np.array(diffuse_col_vec), img_batch.shape)
  print("diffuse.shape = " + str(diffuse.shape))
  return np.clip(diffuse, 0, 1)


def np_phong_composite(image_in, light_dir, light_col, ambient_in,
                       k_diffuse, with_mask=True):
  mask = np_mask(image_in)
  diffuse = np_phong_shading(image_in, light_dir, light_col, k_diffuse)
  if with_mask:
      compos = mask * (ambient_in + diffuse) + (1 - mask)
  else:
      compos = ambient_in + diffuse
  return np.clip(compos, 0, 1)


def generate_light_pos(elevation=90, azimuth=90):
  elevation = (90 - np.array([[elevation]])) * math.pi / 180.0
  azimuth = (np.array([[azimuth]])) * math.pi / 180.0
  x = np.multiply(np.sin(elevation), np.cos(azimuth))
  y = np.multiply(np.sin(elevation), np.sin(azimuth))
  z = np.cos(elevation)
  return np.hstack((x, y, z))


def main():
  fmt_cls = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=fmt_cls)

  parser.add_argument('--voxel_path',
                      type=str,
                      default="./voxel/Misc/bunny.binvox",
                      help="Path to the input voxel.")
  parser.add_argument('--azimuth',
                      type=float,
                      default=250,
                      help="Value of azimuth, between (0,360)")
  parser.add_argument('--elevation',
                      type=float,
                      default=60,
                      help="Value of elevation, between (0,360)")
  parser.add_argument('--light_azimuth',
                      type=float,
                      default=90,
                      help="Value of azimuth for light, between (0,360)")
  parser.add_argument('--light_elevation',
                      type=float,
                      default=90,
                      help="Value of elevation for light, between (0,360)")
  parser.add_argument('--radius',
                      type=float,
                      default=3.3,
                      help="Value of radius, between (2.5, 4.5)")
  parser.add_argument('--render_dir',
                      type=str,
                      default='./render',
                      help='Path to the rendered images.')
  parser.add_argument('--rotate',
                      type=bool,
                      default=False,
                      help=('Flag rotate and render an object by 360 degree in azimuth.'
                            'Overwrites early settings in azimuth.'))
  args = parser.parse_args()

  # We use our "load_graph" function
  graph = load_graph("./model/3d2d_renderer.pb")

  with tf.Session(graph=graph) as sess:
    if not os.path.exists(args.render_dir):
        os.makedirs(args.render_dir)

    azimuth = args.azimuth
    elevation = args.elevation
    radius = args.radius
    light_elevation = args.light_elevation
    light_azimuth = args.light_azimuth
    light_dir = generate_light_pos(light_elevation, light_azimuth)

    with open(args.voxel_path, 'rb') as f:
        voxel = np.reshape(
          binvox_rw.read_as_3d_array(f).data.astype(np.float32),
          (1, 64, 64, 64, 1))
        model_name = os.path.basename(args.voxel_path).split('.binvox')[0]

    if args.rotate:
      # Automatically rorate the object by 360 degree in azimuth
      count = 0
      for azimuth in np.arange(0.0, 360.0, 5.0):
        render(azimuth, elevation, radius, sess, voxel, light_dir,
               args.render_dir, count, light_azimuth, light_elevation,
               model_name)
        count = count + 1
    else:
      # Manually set up pose and light
      render(azimuth, elevation, radius, sess, voxel, light_dir,
             args.render_dir, 0, light_azimuth, light_elevation,
             model_name)


if __name__ == "__main__":
  main()
