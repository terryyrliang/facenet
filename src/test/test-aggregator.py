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
aggregate_topic = config_fetcher.aggregate_topic

producer = KafkaProducer(value_serializer=lambda v:json.dumps(v).encode('utf-8'), bootstrap_servers = servers)

def send_response(response):
    producer.send(aggregate_topic, response)

def send_request(request):
    producer.send(aggregate_topic, request)

print('get ready to send request')
request = eo.AggregatorRequest('cid', 'sid', 'tid', 1, 1, 'E:/git/project/python/facenet/data/images',
                            'E:/git/project/python/facenet/data/images_ext',
                            'E:/git/project/python/facenet/data/images/cid/sid/tid/target/Anthony_Hopkins_0002.jpg',
                            True, 'normal')
jsonObj = eo.convert_to_dict(request);
send_request(jsonObj)

time.sleep(5)