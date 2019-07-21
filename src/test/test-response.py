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
response_topic = config_fetcher.response_topic

producer = KafkaProducer(value_serializer=lambda v:json.dumps(v).encode('utf-8'), bootstrap_servers = servers)


def send_request(value):
    producer.send(response_topic, value=value)

print('get ready to send request')
result = ['abc', 'bbq', 'fds']
response_request = eo.ResponseRequest('cid', 'sid', 'tid', result)
jsonObj = eo.convert_to_dict(response_request);
send_request(jsonObj)

time.sleep(5)