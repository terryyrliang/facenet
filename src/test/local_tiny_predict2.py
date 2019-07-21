import sys

sys.path.append('E:/git/project/facenet/src')
sys.path.append('E:/git/project/facenet/src/align')

import tensorflow as tf
import numpy as np
import facenet
import os
import pickle
import cv2
from api import align_api
import facenet
from datetime import datetime

def print_time_diff(t1, t2):
    print((t2 - t1).seconds)


def perform_predict(image):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            print('load model')
            facenet.load_model(predict_model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            print('load model done')
            with tf.device('/gpu:0'):
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: image, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)
            classifier_filename_exp = os.path.expanduser(classifier_filename2)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
            # print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)
            predictions = model.predict_proba(emb)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            # print predictions
            # print("people in image ")
            # print('%s: %.3f' % (class_names[best_class_indices[k]], best_class_probabilities[k]))
            print(class_names)
            print(best_class_indices)
            print(best_class_probabilities)
            return class_names[best_class_indices[0]]

# classifier_filename = 'E:/git/project/facenet/test-data/test.model'
classifier_filename2 = 'E:/temp/flask-upload-test//test2.model'
predict_model = 'E:/git/project/facenet/model/20180402-114759'
test_image2 = 'E:/temp/flask-upload-test/tmp_n56qh45dbi/raw/Terry/t2.jpg'

images_ = []
img = cv2.imread(test_image2)
img2 = align_api.align_image_data(img)
images_.append(img2)
print(test_image2)
result = perform_predict(images_)
print(result)
