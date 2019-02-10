import numpy as np
import tensorflow as tf
import cv2
import urllib.request

import tfutils.label_map_util as label_map_util
import tfutils.visualization_utils as vis_util

# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
graph_file = 'tfutils/ssd_inception_v2_coco_2017_11_17.pb'
label_file = 'tfutils/mscoco_label_map.pbtxt'
# Number of classes to detect
num_classes = 90
# URL of video feed
URL = 'http://172.16.92.207:8081'

detection_graph = tf.Graph()
category_index = {}

def load_graph(file):
    """Load the detection graph"""
    print('Loading graph')

    
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

def load_label_file(file, num_classes):
    """
    Label maps map indices to category names, so that when our convolution network predicts `5`, 
    we know that this corresponds to `airplane`.  
    Here we use internal utility functions, but anything that returns a dictionary mapping integers
    to appropriate string  label_file would be fine
    TODO: remove dependency on label_map_util
    """
    print('Loading label_file from ' +  label_file)

    label_map = label_map_util.load_labelmap(label_file)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=num_classes, use_display_name=True)
    return label_map_util.create_category_index(categories)



def run(stream):
    """
    Main run loop for the video stream
    """
    print('Detecting')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            stream = urllib.request.urlopen(stream)
            buff = bytes()

            while True:

                # Read frame from camera
                buff += stream.read(1024)
                a = buff.find(b'\xff\xd8') # start code for jpg frame
                b = buff.find(b'\xff\xd9') # end code for jpg frame
                if a != -1 and b != -1:
                    jpg = buff[a:b+2]
                    buff = buff[b+2:]
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    image_np = np.array(img)

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Extract image tensor
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Extract detection boxes
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Extract detection scores
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    # Extract detection classes
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    # Extract number of detectionsd
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=2)

                    # Display output
                    cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break

load_graph(graph_file)
category_index = load_label_file(label_file, num_classes)
run(URL)
