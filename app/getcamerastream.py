# import the opencv library
import cv2
import tensorflow as tf
import numpy as np
# import PIL.Image
from PIL import Image
from datetime import datetime
from urllib.request import urlopen
import cv2

model_filename = 'model.pb'
LABELS_FILENAME = 'labels.txt'

od_model = None
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


  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
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
#     print(i)
        if i["probability"] > 0.25:
            print(i)
            count +=1
            x = int(i["boundingBox"]["left"]*width)
            y = int(i["boundingBox"]["top"]*height)
            w = int(i["boundingBox"]["width"]*width)
            h = int(i["boundingBox"]["height"]*height)
            print(x,y,w,h)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,128), 5)
        # Display the resulting frame
    cv2.putText(frame,str(count), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 3, 255)
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()