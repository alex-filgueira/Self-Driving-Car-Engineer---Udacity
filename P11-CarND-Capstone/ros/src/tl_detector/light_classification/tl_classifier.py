import os
import numpy as np
import tensorflow as tf
import time
import cv2
import datetime

from graph_utils import load_graph
from PIL import ImageDraw, Image
from styx_msgs.msg import TrafficLight

# to save monitor image for debug. if set True, uncomment "import visualization_utils as vis_util" please.
# if "import visualization_utils" occurs error, for detail:https://github.com/MarkBroerkens/CarND-Capstone/blob/master/README.md#prepare-environment
SAVE_MONITOR_IMAGE = False # to save monitor image for debug. 
#import visualization_utils as vis_util

# to show labeled image, for test only, if submit code to udacity, should be set SHOW_MONITOR_IMAGE = False and comment below "imort" lines.  
# take a show: open a new terminal ,and run "rosrun image_view image_view image:=/clssifier_monitor_image"
SHOW_MONITOR_IMAGE = False
#import visualization_utils as vis_util
#import rospy
#from sensor_msgs.msg import Image as Image_msg
#from cv_bridge import CvBridge


class TLClassifier(object):
    def __init__(self, is_site):
        #TODO load classifier

        sess = None

        if is_site:
            sess, _ = load_graph('models/real_model.pb')
        else:
            sess, _ = load_graph('models/sim_model.pb')

        self.sess = sess
        self.sess_graph = self.sess.graph
        # Definite input and output Tensors for sess
        self.image_tensor = self.sess_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.sess_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.sess_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.sess_graph.get_tensor_by_name('detection_classes:0')
        self.num_classes = 3

        self.category_index = {1: {'id': 1, 'name': u'red'}, 2: {'id': 2, 'name': u'yellow'}, 3: {'id': 3, 'name': u'green'}}

        self.image_count = 0
        self.last_pred = TrafficLight.UNKNOWN
        self.pub_tl_clssifier_monitor = None
        self.bridge = None


        if SHOW_MONITOR_IMAGE:
            self.pub_tl_clssifier_monitor = rospy.Publisher('/clssifier_monitor_image', Image_msg, queue_size=2)
            self.bridge = CvBridge()


    def get_classification(self, image, wp = 0):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
        image_np_expanded = np.expand_dims(cv2_image, axis=0)

        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], feed_dict={self.image_tensor: image_np_expanded})

        prediction = 4
        min_score_thresh=.6
        sq_boxes = np.squeeze(boxes)
        sq_classes = np.squeeze(classes).astype(np.int32)
        sq_scores = np.squeeze(scores)

        for i in range(sq_boxes.shape[0]):
            if sq_scores is None or sq_scores[i] > min_score_thresh:
                if sq_classes[i] in self.category_index.keys():
                    prediction = sq_classes[i]
                    print("Found traffic light: {ID:%s  color:%s  pred_score:%.4f}"%(prediction, str(self.category_index[sq_classes[i]]['name']), sq_scores[i]))
                    min_score_thresh = sq_scores[i] 

       
        if SAVE_MONITOR_IMAGE: 

            dt_str = str(time.time()).replace('.','')
            file_name = "Monitor_IMG_" + dt_str + '_'+ str(int(prediction - 1)) +'.jpg'
            output_filename = "light_classification/IMGS/tl_origin/" + file_name
            cv2.imwrite(output_filename, image)

            vis_image = vis_util.visualize_boxes_and_labels_on_image_array(
                     image,
                     sq_boxes,
                     sq_classes,
                     sq_scores,
                     self.category_index,
                     use_normalized_coordinates=True,
                     line_thickness=1
                     #min_score_thresh = min_score_thresh
                     )
            image_pil = Image.fromarray(cv2.cvtColor(vis_image,cv2.COLOR_BGR2RGB))
            file_name = "Monitor_IMG_" + dt_str + '_'+ str(int(prediction - 1)) +'.png'
            output_filename = "light_classification/IMGS/tl_predict_output/" + file_name
            image_pil.save(output_filename, 'PNG')


        if SHOW_MONITOR_IMAGE:
            vis_image = vis_util.visualize_boxes_and_labels_on_image_array(
                     image,
                     sq_boxes,
                     sq_classes,
                     sq_scores,
                     self.category_index,
                     use_normalized_coordinates=True,
                     line_thickness=1
                     )

            image_message = self.bridge.cv2_to_imgmsg(cv2.cvtColor(vis_image,cv2.COLOR_BGR2RGB), encoding="rgb8")
            self.pub_tl_clssifier_monitor.publish(image_message)


        rtn = TrafficLight.UNKNOWN

        if prediction == 1:
            rtn = TrafficLight.RED
        elif prediction == 2:
            rtn = TrafficLight.YELLOW
        elif prediction == 3:
            rtn = TrafficLight.GREEN

        self.last_pred = rtn
        self.image_count += 1

        return rtn