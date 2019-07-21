import os
import shutil
import random
import string
from ctypes import cdll
import time
import datetime

print('init utils.py')

def check_detect_lock(lock_path):
    if os.path.isfile(lock_path + '.lock'):
        return True
    return False

def acquire_detect_lock(lock_path):
    if check_clf_lock(lock_path):
        return False
    file = open(lock_path + '.lock', 'w')
    file.close()
    return True

def release_detect_lock(lock_path):
    if (os.path.isfile(lock_path + '.lock')):
        os.remove(lock_path + '.lock')
##################################################################
def acquire_lock(root_path, label):
    if check_clf_lock(root_path, label):
        return False
    file = open(root_path + '/' + label + '.lock', 'w')
    file.close()
    return True

def release_lock(root_path, label):
    if (os.path.isfile(root_path + '/' + label + '.lock')):
        os.remove(root_path + '/' + label + '.lock')

def check_clf_lock(root_path, label):
    if os.path.isfile(root_path + '/' + label + '.lock'):
        return True
    return False

def generate_done_file(root_path, label):
    os.mknod(root_path + '/' + label + '.done')

def remove_done_file(root_path, label):
    if (os.path.isfile(root_path + '/' + label + '.done')):
        os.remove(root_path + '/' + label + '.done')

def build_collection_id(user_id, label):
    return user_id + '_' + label

def makeup_images(target_path):
    src_path = 'E:/temp/flask-upload-test/Unknown'
    if not os.path.isdir(target_path + '/' + src_path.split("/")[-1]):
        shutil.copytree(src_path, target_path + '/' + src_path.split("/")[-1])

def get_random_str():
    return ''.join(random.sample(string.ascii_letters + string.digits, 28))


_sopen = cdll.msvcrt._sopen
_SH_DENYRW = 0x10

def is_open(filename):
    if not os.access(filename, os.F_OK):
        return False # file doesn't exist
    h = _sopen(filename, 0, _SH_DENYRW, 0)
    if h == 3:
        return False # file is not opened by anyone else
    return True # file is already open

def get_timebased_imagename():
    seed = ''.join(random.sample(string.ascii_letters + string.digits, 10))
    time1 = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S_") + seed
    return time1

def get_curtime():
    time1 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return time1

# src_path = 'E:/temp/flask-upload-test/Aaron_Sorkin'
# target_path = 'E:/temp/flask-upload-test/tmp_11ador4gyg2c_Terry/raw'
# shutil.copytree(src_path, target_path + '/' + src_path.split("/")[-1])