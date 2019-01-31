# TF_SMPL
  TF_SMPL is a TensorFlow implementation of [Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image](http://smplify.is.tue.mpg.de/), it can estimate the 3D pose and shape of the human body from a single image.

# Dependencies
  This code uses Python 2.7 and need the following dependencies, I recommend you to use a virtual environment.  
  * numpy, tensorflow-cpu, matplotlib, tqdm, time, cv2, pickle, opendr(if you want to render the result), scipy.

# Usage
  Download the model, (http://smpl.is.tue.mpg.de/downloads).  
  Run gensh.sh, and then run generate_video.py to generate a video.  
    `sh gensh.sh`  
    `python generate_video.py`  
  If you want to use your own data, you need provide image and the corresponding 2D joints.  
  You can use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) or something similar to get 2D joints.  
  Please find more details in this code.
