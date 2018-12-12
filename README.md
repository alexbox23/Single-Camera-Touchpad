# Single-Camera-Touchpad

This project uses a single camera to convert an ordinary surface into a touch interface. Touch is registered by monitoring the user's blanch response. The color of the user's finger appears lighter when the finger is applying pressure on a surface.

Detailed descriptions can be found in report.pdf. Click on the images below to see demo videos.

[![smiley](http://img.youtube.com/vi/ZdTUgK25fyQ/0.jpg)](https://youtu.be/ZdTUgK25fyQ)
[![hi](http://img.youtube.com/vi/iNe2k8-xOqs/0.jpg)](https://youtu.be/iNe2k8-xOqs)

Dependencies:
>Python3 (3.6)\
>numpy (1.14)\
>TensorFlow (1.12)\
>OpenCV (3.4)\
>imutils (0.5)

To run fingertip_labeller.py, download the [11K Hands dataset](https://sites.google.com/view/11khands) and unzip in the 11k_hands folder. The labeller script will write to 11k_hands/fingertip_labels.csv and can resume from the last labelled image using the option --resume.

Run csv_to_tfrecord.py to convert the label data into TFRecord format. This will generate 10 shards each for training and evaluation: fingertips_train.record-0000x-of-00010 and fingertips_val.record-0000x-of-00010 

In order to train the neural network, clone the repo [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/object_detection) and follow the installation instructions. The files needed for training, as well as the latest model trained for this project, are saved in this repo under the folder tensorflow. 

The main touch detection functionality is split between two scripts: track_fingertips.py and generate_touch_path.py. The homography and touch path visualization are separated from the actual detection and tracking because I could not get the framerate higher than 4 fps on the Raspberry Pi. A pre-recorded video can be used instead of the video stream from the Raspberry Pi. 

Pull-requests are welcome. Some possible optimizations are mentioned in report.pdf.