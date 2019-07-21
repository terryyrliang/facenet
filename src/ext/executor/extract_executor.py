import os, sys
app_path = os.environ['APP_PATH']
for p in app_path.split(';'):
    sys.path.append(p)

import dlib
from skimage import io
from common import config_fetcher
from kafka import KafkaConsumer
from kafka import KafkaProducer
from bean import event
import json

root_dir = "E:/git/project/python/facenet/data/images/"
new_dir = "E:/git/project/python/facenet/data/images_ext"
sufix = "_extract_x"

servers = config_fetcher.bootstrap_hosts
group_id = config_fetcher.group_id
compare_topic = config_fetcher.compare_topic
extract_topic = config_fetcher.extract_topic

model = config_fetcher.model
# To consume latest messages and auto-commit offsets
consumer = KafkaConsumer(extract_topic,
                         group_id = group_id,
                         bootstrap_servers = servers,
                         value_deserializer=lambda m: json.loads(m.decode('ascii')))

producer = KafkaProducer(value_serializer=lambda v:json.dumps(v).encode('utf-8'), bootstrap_servers = servers)

detector = dlib.get_frontal_face_detector()

def execute():
    for message in consumer:
        request = message.value
        client_id = request['client_id']
        session_id = request['session_id']
        trace_id = request['trace_id']
        root_path = request['root_path']
        dataset_root_path = request['dataset_root_path']
        target_image_path = request['target_image_path']

        target_ex_path = extract_faces(target_image_path)[0]

        face_image_paths = []
        scan_root_path(dataset_root_path, face_image_paths)
        print('print scanning images')
        total_image_num = len(face_image_paths)
        order = 1
        for fip in face_image_paths:
            print(fip)
            face_ex_path = extract_faces(fip)
            compare_request = build_next_request(client_id, session_id, trace_id, total_image_num, order, root_path, new_dir,
                                                 target_image_path, face_ex_path, target_ex_path)
            order = order + 1
            print('-----------------------------------')
            print(event.convert_to_dict(compare_request))
            producer.send(compare_topic, compare_request)
            print('-----------------------------------')


def build_next_request(client_id, session_id, trace_id, total_image_numbers, request_order, root_path, extract_root_path,
                       face_image_path, face_extract_path, target_extract_path):
    compare_request = event.CompareRequest(client_id, session_id, trace_id, total_image_numbers,
                                           request_order, root_path, extract_root_path, face_image_path, face_extract_path, target_extract_path)
    return compare_request

def extract_faces(file):
    face_ex_path = []
    file = file.replace("\\", "/")
    print("Get ready to process file " + file)
    img = io.imread(file)
    try:
        faces = detector(img, 1)
    except Exception as e:
        print('No face detect for image ' + file)
        return face_ex_path
    print(file + " faces in all：", len(faces))

    for i, d in enumerate(faces):
        bname = os.path.basename(file)
        out = os.path.splitext(bname)
        ext = ".jpg"
        if (len(out) == 2):
            ext = out[1]
        target_dir = file.replace(root_dir, new_dir + "/")
        if (file.endswith("/")):
            target_dir = file[:-1]
        target_dir = target_dir + sufix
        if (not os.path.exists(target_dir)):
            os.makedirs(target_dir)
        target_path = target_dir + "/" + bname + "_" + str(i) + ext
        print(target_path)
        face_ex_path.append(target_path)
        io.imsave(target_path, img[d.top():d.bottom(), d.left():d.right()])
    return face_ex_path

def scan_root_path(dir, array):
    if (os.path.isdir(dir)):
        dirs = os.listdir(dir)
        for file in dirs:
            abs_path = os.path.join(dir, file)
            if (os.path.isdir(abs_path)):
                scan_root_path(abs_path, array)
            else:
                array.append(abs_path)
    else :
        array.append(dir)

if __name__ == "__main__":
    execute()