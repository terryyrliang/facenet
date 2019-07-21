from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")
sys.path.append('E:/git/project/facenet/src')
sys.path.append('E:/git/project/facenet/src/align')
import tensorflow as tf
import numpy as np
import argparse
import os
import math
import pickle
from sklearn.svm import SVC
from api import align_api
import facenet
from datetime import datetime

def print_time_diff(t1, t2):
    print((t2 - t1).seconds)

def mainProcess(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            np.random.seed(seed=args.seed)
            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class,
                                                    args.nrof_train_images_per_class)
                if (args.mode == 'TRAIN'):
                    dataset = train_set
                elif (args.mode == 'CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(args.data_dir)
            # Check that there are at least one training image per class
            for cls in dataset:
                assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

            with tf.device('/gpu:0'):
                paths, labels = facenet.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            # Load the model
            print('Loading feature extraction model')
            t1 = datetime.now()
            facenet.load_model(args.model)

            t2 = datetime.now()
            print_time_diff(t1, t2)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            with tf.device('/gpu:0'):
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            t3 = datetime.now()
            print_time_diff(t2, t3)
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            t4 = datetime.now()
            print_time_diff(t3, t4)
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                t5 = datetime.now()
                print_time_diff(t4, t5)
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                t6 = datetime.now()
                print_time_diff(t5, t6)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                t7 = datetime.now()
                print_time_diff(t6, t7)
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
                t8 = datetime.now()
                print_time_diff(t7, t8)

            t9 = datetime.now()
            print_time_diff(t8, t9)
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)
            t10 = datetime.now()
            print_time_diff(t9, t10)

            if (args.mode == 'TRAIN'):
                # Train classifier
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                t11 = datetime.now()
                print_time_diff(t10, t11)
                model.fit(emb_array, labels)
                t12 = datetime.now()
                print_time_diff(t11, t12)
                # Create a list of class names
                class_names = [cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                t13 = datetime.now()
                print_time_diff(t12, t13)
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                t14 = datetime.now()
                print_time_diff(t13, t14)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

            elif (args.mode == 'CLASSIFY'):
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)


def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths) >= min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
                        help='Indicates if a new classifier should be trained or a classification ' +
                             'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename',
                        help='Classifier model file name as a pickle (.pkl) file. ' +
                             'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset',
                        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +
                             'Otherwise a separate test set can be specified using the test_data_dir option.',
                        action='store_true')
    parser.add_argument('--test_data_dir', type=str,
                        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
                        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
                        help='Use this number of images from each class for training and the rest for testing',
                        default=10)

    return parser.parse_args(argv)

sess_dir = 'E:/temp/flask-upload-test/TFnEYmC19RI8Wtgr3OGpXd7Swlxy'
align_api.opimize_physical_images(sess_dir)
argv = []
argv.append('TRAIN')
argv.append(sess_dir + '/160/')
argv.append('E:/git/project/facenet/model/20180402-114759')
argv.append('E:/temp/flask-upload-test/Full.model')
print(argv)
if (os.path.isfile('E:/temp/flask-upload-test/Full.model')):
    os.remove('E:/temp/flask-upload-test/Full.model')

t1 = datetime.now()
mainProcess(parse_arguments(argv[0:]))
t2 = datetime.now()
print((t2 - t1).seconds)

# if __name__ == '__main__':
#     main(parse_arguments(sys.argv[1:]))