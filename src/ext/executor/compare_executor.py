from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf
from scipy import misc

app_path = os.environ['APP_PATH']
for p in app_path.split(';'):
    sys.path.append(p)
import os
import copy
import facenet
import align.detect_face
from common import config_fetcher
from kafka import KafkaConsumer
from kafka import KafkaProducer
import json
from bean import event

servers = config_fetcher.bootstrap_hosts
group_id = config_fetcher.group_id
compare_topic = config_fetcher.compare_topic
aggregate_topic = config_fetcher.aggregate_topic

model = config_fetcher.model
# To consume latest messages and auto-commit offsets
consumer = KafkaConsumer(compare_topic,
                         group_id = group_id,
                         bootstrap_servers = servers,
                         value_deserializer=lambda m: json.loads(m.decode('ascii')))

producer = KafkaProducer(value_serializer=lambda v:json.dumps(v).encode('utf-8'), bootstrap_servers = servers)

def execute():
    # images = load_and_align_data(image_files, image_size, margin, gpu_memory_fraction)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(model)
            #########################################################################################################
            ############################################# Split Line ################################################
            #########################################################################################################
            print('load model done...')
            for message in consumer:
                try:
                    request = message.value
                    image_files = request['face_extract_path']
                    target_extract_path = request['target_extract_path']
                    image_files.insert(0, target_extract_path)
                    print("get a request")
                    images = load_and_align_data(image_files, config_fetcher.compare_is, config_fetcher.compare_margin, config_fetcher.compare_gmf)
                    # Get input and output tensors
                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                    # Run forward pass to calculate embeddings
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb = sess.run(embeddings, feed_dict=feed_dict)

                    nrof_images = len(image_files)

                    print_target_images(nrof_images, image_files)
                    print_result_matrix(np, nrof_images, emb)
                    fr = extract_final_result(np, nrof_images, emb)
                    result = False
                    for r in fr:
                        if r < 1:
                            result = True
                            break
                    next_request = build_next_request(request, result, '')
                    print('-----------------------------------')
                    print(event.convert_to_dict(next_request))
                    producer.send(aggregate_topic, next_request)
                    print('-----------------------------------')
                    print("process one request done...")
                except Exception as e:
                    print(e)

def extract_final_result(np, nrof_images, emb):
    final_result = []
    for j in range(1, nrof_images):
        dist = np.sqrt(np.sum(np.square(np.subtract(emb[0, :], emb[j, :]))))
        final_result.append(dist)
    return final_result

def print_target_images(nrof_images, image_files):
    print('Images:')
    for i in range(nrof_images):
        print('%1d: %s' % (i, image_files[i]))
    print('')

def print_result_matrix(np, nrof_images, emb):
    # Print distance matrix
    print('Distance matrix')
    print('    ', end='')
    for i in range(nrof_images):
        print('    %1d     ' % i, end='')
    print('')
    for i in range(nrof_images):
        print('%1d  ' % i, end='')
        for j in range(nrof_images):
            dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
            print('  %1.4f  ' % dist, end='')
        print('')

def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    tmp_image_paths = copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            print("can't detect face, remove ", image)
            continue
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

def build_next_request(old_request, result, error_message):
    nr = event.AggregatorRequest(old_request['client_id'], old_request['session_id'], old_request['trace_id'],
                                 old_request['total_image_numbers'], old_request['request_order'], old_request['root_path'],
                                 old_request['extract_root_path'], old_request['face_image_path'], result, error_message)
    return nr


if __name__ == "__main__":
    execute()