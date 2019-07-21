import concurrent.futures
from service import train_image_service as ti_service

running_id = {}

pool = concurrent.futures.ThreadPoolExecutor(5)

def submit_train_task(root_path, label):
    print('Call submit_train_task')
    if (not label in running_id or running_id[label] != 'running'):
        running_id[label] = 'running'
        pool.submit(process_train, root_path, label)
    else:
        print('%s is running' % label)

def process_train(root_path, label):
    print('Run process_train task %s' % label)
    ti_service.train_image(root_path, label)
    running_id[label] = 'done'
    print('Done process_train task %s' % label)
