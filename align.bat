@echo off
set PYTHONPATH=E:\git\project\python\facenet\src
set PROJECT_PATH=E:/git/project/python/facenet
python %PROJECT_PATH%/src/align/align_dataset_mtcnn.py %PROJECT_PATH%/data/images %PROJECT_PATH%/data/imagesa_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25