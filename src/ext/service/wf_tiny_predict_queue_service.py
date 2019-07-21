import sys
sys.path.append('E:/git/project/facenet/src')
sys.path.append('E:/git/project/facenet/src/align')

import tensorflow as tf
import numpy as np
import facenet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import cv2
from api import align_api
import facenet
from datetime import datetime
import threading
import queue
from configuration import configuration_service
from utils import utils
import datetime

predict_event_queue = queue.Queue()
predict_manager = {}

class PredictRequest():
    def __init__(self, user_id, label, image_file_name, image_data, classifier_filename):
        self.user_id = user_id
        self.label = label
        self.image_file_name = image_file_name
        self.image_data = image_data
        self.classifier_filename = classifier_filename

class PredictResponse():
    def __init__(self, image_file_name, classifier_filename, owner):
        self.image_file_name = image_file_name
        self.classifier_filename = classifier_filename
        self.owner = owner

class PredictObject():
    def __init__(self, collection_id, predict_request, predict_response):
        self.collection_id = collection_id
        self.predict_request = predict_request
        self.predict_response = predict_response

def build_predict_request(user_id, label, cls_root_path, target_file):
    image_file_path = target_file
    classifier_filename = cls_root_path + '/' + label + configuration_service.classify_file_extention
    images = []
    img = cv2.imread(image_file_path)
    if img.any() != None:
        img2 = align_api.align_image_data(img)
        images.append(img2)
        return PredictRequest(user_id, label, image_file_path, images, classifier_filename)

def handle_prediction_completion(predict_request, predict_result_name):
    print('Call handle_predictino_completion')
    collection_id = utils.build_collection_id(predict_request.user_id, predict_request.label)
    if collection_id in predict_manager:
        p_object = predict_manager[collection_id]
        p_request = p_object.predict_request
        p_response = PredictResponse(p_request.image_file_name, p_request.classifier_filename, predict_result_name)
        p_object.predict_response = p_response

def can_enqueue(user_id, label):
    collection_id = utils.build_collection_id(user_id, label)
    if collection_id in predict_manager:
        # This is not the first time to access
        p_object = predict_manager[collection_id]
        if p_object == None:
            return True
        if  not hasattr(p_object, 'predict_response') or p_object.predict_response == None:
            # The prediction is still ongoing
            return False
        else:
            # This prediction is done
            return True
    # This is the first time to access
    return True

def dequeue_predict(user_id, label, cls_root_path, target_file):
    collection_id = utils.build_collection_id(user_id, label)
    if collection_id in predict_manager:
        # This is not the first time to access
        predict_manager.pop(collection_id)

# Entrance for the prediction
def enqueue_predict(user_id, label, cls_root_path, target_file):
    collection_id = utils.build_collection_id(user_id, label)
    if collection_id in predict_manager:
        # This is not the first time to access
        p_object = predict_manager[collection_id]
        if not hasattr(p_object, 'predict_response') or p_object.predict_response == None:
            # The prediction is still ongoing
            return None
        else:
            # This prediction is done
            return p_object.predict_response
    else:
        # This is the first time to access
        p_request = build_predict_request(user_id, label, cls_root_path, target_file)
        if p_request != None:
            p_object = PredictObject(collection_id=collection_id, predict_request=p_request, predict_response=None)
            predict_manager[collection_id] = p_object
            predict_event_queue.put(p_request)

class PredictThread(threading.Thread):
    def __init__(self):
        super(PredictThread, self).__init__()

    def run(self):
        print('Start to process predict request')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=configuration_service.gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                t1 = datetime.datetime.now()
                print('Load model - ' + utils.get_curtime())
                with tf.device('/cpu:0'):
                    facenet.load_model(configuration_service.pretrain_model_path)
                print('Load model done')
                t2 = datetime.datetime.now()
                print((t2 - t1).seconds)
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                with tf.device('/gpu:0'):
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                #########################################################################################################
                ############################################# Split Line ################################################
                #########################################################################################################
                while True:
                    p_request = predict_event_queue.get(5)
                    try:
                        if p_request != None:
                            print('Start to predict : ', p_request.image_file_name)
                            images = p_request.image_data
                            c_filename = select_classifier_file(p_request.classifier_filename)
                            print('Use classifier file : ', c_filename)
                            if c_filename != None:
                                # Run forward pass to calculate embeddings
                                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                                emb = sess.run(embeddings, feed_dict=feed_dict)
                                classifier_filename_exp = os.path.expanduser(c_filename)
                                if not os.path.isfile(classifier_filename_exp):
                                    print('Not train the image yet for %s' % c_filename)
                                    handle_prediction_completion(p_request, 'Unknown')
                                else:
                                    with open(classifier_filename_exp, 'rb') as infile:
                                        (model, class_names) = pickle.load(infile)
                                    # print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)
                                    predictions = model.predict_proba(emb)
                                    best_class_indices = np.argmax(predictions, axis=1)
                                    best_class_probabilities = predictions[
                                        np.arange(len(best_class_indices)), best_class_indices]
                                    print(class_names)
                                    print(best_class_indices)
                                    print(best_class_probabilities)
                                    pre_name = 'Unknown'
                                    if len(best_class_indices) > 0:
                                        pre_name = class_names[best_class_indices[0]]
                                    print('The best result is : ' + pre_name)
                                    handle_prediction_completion(p_request, pre_name)
                                    # return class_names[best_class_indices[0]]
                            else:
                                print('Not tran image yet for %s' % p_request.classifier_filename)
                    except Exception:
                        try:
                            handle_prediction_completion(p_request, 'Unknown')
                            print(Exception.message)
                        except Exception:
                            print('second exception')
                sess.close()
                print('out of the while loop')

def select_classifier_file(custom_classifier_file):
    # if not os.path.isfile(custom_classifier_file):
    #     return configuration_service.general_classifier_file
    # return custom_classifier_file
    return configuration_service.general_classifier_file