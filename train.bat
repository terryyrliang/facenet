@echo off
set PYTHONPATH=E:\git\project\facenet\src
python src/classifier.py TRAIN E:/git/project/facenet/data/lfw/raw2_160 E:/git/project/facenet/model/20180402-114759 E:/git/project/facenet/test-data/abc
