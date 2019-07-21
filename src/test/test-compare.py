import os
import sys

app_path = os.environ['APP_PATH']
for p in app_path.split(';'):
    sys.path.append(p)

from common import config_fetcher
from kafka import KafkaProducer
import json
import time
from bean import event as eo

servers = config_fetcher.bootstrap_hosts
group_id = config_fetcher.group_id
compare_topic = config_fetcher.compare_topic

producer = KafkaProducer(value_serializer=lambda v:json.dumps(v).encode('utf-8'), bootstrap_servers = servers)

def send_response(response):
    producer.send(compare_topic, response)

def send_request(request):
    producer.send(compare_topic, request)

print('get ready to send request')
request = eo.CompareRequest('cid', 'sid', 'tid', 9, 9, 'E:/git/project/python/facenet/data/images',
                            'E:/git/project/python/facenet/data/images_ext',
                            'E:/git/project/python/facenet/data/images/cid/sid/tid/target/Anthony_Hopkins_0002.jpg',
                            ['E:/git/project/python/facenet/data/images_ext/cid/sid/tid/dataset/federa/5.jpg_extract_x/5.jpg_0.jpg',
                             'E:/git/project/python/facenet/data/images_ext/cid/sid/tid/dataset/Anthony_Hopkins/Anthony_Hopkins_0001.jpg_extract_x/Anthony_Hopkins_0001.jpg_0.jpg',
                             'E:/git/project/python/facenet/data/images_ext/cid/sid/tid/dataset/federa/4.jpg_extract_x/4.jpg_0.jpg',
                             'E:/git/project/python/facenet/data/images_ext/cid/sid/tid/dataset/federa/4.jpg_extract_x/4.jpg_1.jpg'],
                            'E:/git/project/python/facenet/data/images_ext/cid/sid/tid/target/Anthony_Hopkins_0002.jpg_extract_x/Anthony_Hopkins_0002.jpg_0.jpg')
jsonObj = eo.convert_to_dict(request);
send_request(jsonObj)

time.sleep(5)