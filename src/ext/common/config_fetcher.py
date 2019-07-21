#############################################
### web face setting
#############################################
pretrain_model_path = 'E:/git/project/facenet/model/20180402-114759'
train_mode = 'TRAIN'
classify_mode = 'CLASSIFY'
profile_data_root_path = 'E:/temp/flask-upload-test/'
image_upload_path =  'E:/temp/flask-upload-test/'
detect_dir = 'detect'
post_process_image_size = 160
gpu_memory_fraction = 0.25
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
margin = 44
detect_multiple_faces = True

use_split_dataset = False
seed = 666
nrof_train_images_per_class = 10
min_nrof_images_per_class = 20
batch_size = 90

predict_model_running = False

image_raw_dir = 'raw'

classify_file_extention = '.model'
predict_image_extention = '.png'
training_file_extention = '.training'

general_classifier_file = 'E:/temp/flask-upload-test/Full.model'

###############################################
### face search setting
###############################################
## KAFKA settings
bootstrap_hosts = ["localhost:9092"]
extract_topic = "extract_request"
compare_topic = "compare_topic"
aggregate_topic = "aggregate_topic"
response_topic = "response_topic"
group_id = "event_processor"

model = 'E:/git/project/python/facenet/model/20180402-114759'

# image size
compare_is = 160
#
compare_margin = 44
#gpu_memory_fraction
compare_gmf = 1.0