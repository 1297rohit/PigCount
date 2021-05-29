from picamera.array import PiRGBArray # Generates a 3D RGB array
from picamera import PiCamera # Provides a Python interface for the RPi Camera Module
import time # Provides time-related functions
import cv2
import tensorflow as tf
import numpy as np
from azure.iot.device import IoTHubDeviceClient, Message

#azure connection string
CONNECTION_STRING = "HostName=HUBDEVKIT.azure-devices.net;DeviceId=piazuretest;SharedAccessKey=L000kRCO4zda4rzBp6WMJGAn4Wbpanscdj2jChib0WQ="
def iothub_client_init():
    # Create an IoT Hub client
    client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
    return client

MSG_TXT = '{{"pigcount": {count}}}'

#model files
model_filename = 'model.pb'
LABELS_FILENAME = 'labels.txt'
labels = None
INPUT_TENSOR_NAME = 'image_tensor:0'
OUTPUT_TENSOR_NAMES = ['detected_boxes:0', 'detected_scores:0', 'detected_classes:0']

graph_def = tf.compat.v1.GraphDef()
with open(model_filename, 'rb') as f:
    graph_def.ParseFromString(f.read())

graph = tf.Graph()
with graph.as_default():
    tf.import_graph_def(graph_def, name='')

# Get input shape
with tf.compat.v1.Session(graph=graph) as sess:
    input_shape = sess.graph.get_tensor_by_name(INPUT_TENSOR_NAME).shape.as_list()[1:3]


with open(LABELS_FILENAME) as f:
        labels = [l.strip() for l in f.readlines()]


def predict_image(image):
        inputs = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        with tf.compat.v1.Session(graph=graph) as sess:
            output_tensors = [sess.graph.get_tensor_by_name(n) for n in OUTPUT_TENSOR_NAMES]
            outputs = sess.run(output_tensors, {INPUT_TENSOR_NAME: inputs})
            return outputs  




class VideoCamera(object):
    def __init__(self):
        self.camera = PiCamera()
        self.camera.resolution = (640, 480)
        self.camera.framerate = 32
        self.raw_capture = PiRGBArray(self.camera, size=(640, 480))
        self.client = iothub_client_init()
        time.sleep(0.1)

    def __del__(self):
        self.raw_capture.truncate(0)

    def get_frame(self,min_conf_threshold):
        for image in self.camera.capture_continuous(self.raw_capture, format="bgr", use_video_port=True):
            frame = image.array
            height,width,_ = frame.shape
            screen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(screen, (tuple(input_shape)), interpolation = cv2.INTER_AREA)
            pred_out = predict_image(resized)
            predictions = [{'probability': round(float(p[1]), 8),
                            'tagId': int(p[2]),
                            'tagName': labels[p[2]],
                            'boundingBox': {
                                'left': round(float(p[0][0]), 8),
                                'top': round(float(p[0][1]), 8),
                                'width': round(float(p[0][2] - p[0][0]), 8),
                                'height': round(float(p[0][3] - p[0][1]), 8)
                                }
                            } for p in zip(*pred_out)]
            count = 0 
            for i in predictions:
                if i["probability"] > min_conf_threshold and i["tagName"] == "pig":
                    count +=1
                    x = int(i["boundingBox"]["left"]*width)
                    y = int(i["boundingBox"]["top"]*height)
                    w = int(i["boundingBox"]["width"]*width)
                    h = int(i["boundingBox"]["height"]*height)
                    # print(x,y,w,h)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,128), 5)
            cv2.putText(frame,str(count),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
            msg_txt_formatted = MSG_TXT.format(count=count)
            message = Message(msg_txt_formatted)
            print( "Sending message: {}".format(message) )
            self.client.send_message(message)
            print ( "Message successfully sent" )
            time.sleep(5)
            ret, jpeg = cv2.imencode('.jpg', frame)
            self.raw_capture.truncate(0)

            return jpeg.tobytes()