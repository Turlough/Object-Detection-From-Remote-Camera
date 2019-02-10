# Tensorflow object identification
This works well with [motioneye](https://randomnerdtutorials.com/install-motioneyeos-on-raspberry-pi-surveillance-camera-system/) 
software or OS running on a Raspberry Pi on your home network. The Pi provides the remote video stream.
This code runs on your PC, with a graphics card, and identifies objects on the remote stream.
You will need to edit the IP address on line 15 of _stream detector.py_

# How to use:
* Change the IP address and port on line 15 of _stream detector.py_ to match that of your IP camera.
* To install packages indicated in _requirements.txt_, run command `pip install -r requirements.txt`. 
You may want to create a [virtual environment](https://docs.python-guide.org/dev/virtualenvs/) for this.
You will only have to do this once.
* If using a virtual environment, activate it first: `source venv/bin/activate`
* `python stream_detector.py` will run object recogition on the remote stream.

## Acknowledgments
Relies heavily on code extracted from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md.
The improvement here is use of a remote video stream instead of an onboard or usb camera, and some other minor changes.
