import os
import sys

app_path = os.environ['APP_PATH']
for p in app_path.split(';'):
    sys.path.append(p)
from common import config_fetcher
from kafka import KafkaConsumer
from kafka import KafkaProducer
import aggregator_storage_service as ass
import json

servers = config_fetcher.bootstrap_hosts
group_id = config_fetcher.group_id
aggregate_topic = config_fetcher.aggregate_topic
response_topic = config_fetcher.response_topic

model = config_fetcher.model
# To consume latest messages and auto-commit offsets
consumer = KafkaConsumer(aggregate_topic,
                         group_id = group_id,
                         bootstrap_servers = servers,
                         value_deserializer=lambda m: json.loads(m.decode('ascii')))

producer = KafkaProducer(value_serializer=lambda v:json.dumps(v).encode('utf-8'), bootstrap_servers = servers)

def execute():
    for message in consumer:
        request = message.value
        print(request)
        client_id = request['client_id']
        session_id = request['session_id']
        trace_id = request['trace_id']
        total_numbers = request['total_image_numbers']
        root_path = request['root_path']
        result = request['result']

        face_image_path = request['face_image_path']
        face_image_path = face_image_path.replace(root_path, "")
        face_image_path = face_image_path.replace(os.path.join(client_id, session_id, trace_id).replace("\\", "/"), "")
        key = ass.format_key(client_id, session_id, trace_id)
        ass.put2(key, face_image_path)
        if result:
            ass.put(key, face_image_path)
        num = ass.get_numbers(key)
        if (num >= total_numbers):
            producer.send(response_topic, key, ass.extract_result(key))
            ass.print_result(key)
        else:
            print('not receive all the result yet for {}', key)

if __name__ == "__main__":
    execute()