from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('E:/git/project/facenet/src')
import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import math
import pickle
from sklearn.svm import SVC
from api import align_api
from configuration import configuration_service
from utils import utils
import queue
import threading
import datetime

train_event_queue = queue.Queue()

class TrainRequest():
    def __init__(self, image_root_path, label):
        self.image_root_path = image_root_path
        self.label = label

def build_train_request(root_dir, label):
    return TrainRequest(root_dir, label)

def handle_train_completion(training_file):
    if os.path.isfile(training_file):
        os.remove(training_file)

def can_enqueue(image_root_path, label):
    training_file = image_root_path + '/' + label + '/' + configuration_service.training_file_extention
    return not os.path.isfile(training_file)

# Entrance for the train
# root_dir is <root_path>/<user_id>_<label>
def enqueue_train(root_dir, label):
    training_file = root_dir + '/' + label + configuration_service.training_file_extention
    if not os.path.isfile(training_file):
        file = open(training_file, 'w')
        file.close()
    train_event_queue.put(build_train_request(root_dir, label))
        
##################################################################

class TrainThread(threading.Thread):
    def __init__(self):
        super(TrainThread, self).__init__()

    def run(self):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                # Load the model
                t1 = datetime.datetime.now()
                print('Train - Loading feature extraction model')
                with tf.device('/cpu:0'):
                    facenet.load_model(configuration_service.pretrain_model_path)
                print('Train - Loading feature extraction model done')
                t2 = datetime.datetime.now()
                print((t2 - t1).seconds)
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                with tf.device('/cpu:0'):
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
                #########################################################################################################
                ############################################# Split Line ################################################
                #########################################################################################################
                while True:
                    try:
                        t_request = train_event_queue.get(5)
                        if t_request != None:
                            print('Start to train : ', t_request.label)
                            image_root_path = t_request.image_root_path
                            label = t_request.label
                            utils.makeup_images(image_root_path + '/' + configuration_service.image_raw_dir)
                            align_api.opimize_physical_images(t_request.image_root_path)
                            classifier_filename = image_root_path + '/' + label + configuration_service.classify_file_extention
                            data_dir = t_request.image_root_path + '/' + str(configuration_service.post_process_image_size)

                            np.random.seed(seed=configuration_service.seed)
                            dataset = facenet.get_dataset(data_dir)

                            with tf.device('/cpu:0'):
                                paths, labels = facenet.get_image_paths_and_labels(dataset)

                            print('Number of classes: %d' % len(dataset))
                            print('Number of images: %d' % len(paths))
                            # Run forward pass to calculate embeddings
                            print('Calculating features for images')
                            nrof_images = len(paths)
                            with tf.device('/gpu:0'):
                                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / configuration_service.batch_size))
                            with tf.device('/gpu:0'):
                                emb_array = np.zeros((nrof_images, embedding_size))
                            for i in range(nrof_batches_per_epoch):
                                start_index = i * configuration_service.batch_size
                                end_index = min((i + 1) * configuration_service.batch_size, nrof_images)
                                paths_batch = paths[start_index:end_index]
                                images = facenet.load_data(paths_batch, False, False, configuration_service.post_process_image_size)
                                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                            classifier_filename_exp = os.path.expanduser(classifier_filename)

                            # Train classifier
                            print('Training classifier')
                            model = SVC(kernel='linear', probability=True)

                            model.fit(emb_array, labels)

                            # Create a list of class names
                            class_names = [cls.name.replace('_', ' ') for cls in dataset]

                            # Saving classifier model
                            with open(classifier_filename_exp, 'wb') as outfile:
                                pickle.dump((model, class_names), outfile)
                            print('Saved classifier model to file "%s"' % classifier_filename_exp)
                            training_file = image_root_path + '/' + label + configuration_service.training_file_extention
                            handle_train_completion(training_file)
                    except Exception:
                        print(Exception.message)
                print('Leave training')
                sess.close()
