# -*- coding: utf-8 -*-
"""
程式功能: 檢查自己的tenserflow 是否有用GPU環境運行。
一般安裝步驟: https://tf.wiki/zh_hant/basic/installation.html
use
pip install tensorflow
pip install keras
download
(from tensorflow 2.1, we don't need "pip install tensorflow-gpu" for GPU)

安裝tensorflow若失敗可能可看底下這篇
https://tn00343140a.pixnet.net/blog/post/316145031-%E5%AE%89%E8%A3%9Dtensorflow%E6%99%82%2C-%E9%81%87%E5%88%B0error%3A-cannot-uninstall-%27wrapt%27.-


for download GPU, use:
    
conda install cudatoolkit=10.1
conda install cudnn=7.6.5
"""
import keras
import tensorflow
from tensorflow.python.client import device_lib
for d in device_lib.list_local_devices():
    print(d.name, d.device_type, d.physical_device_desc)

"""
my version:
keras version: 2.4.3
tensorflow version: 2.4.0
"""
print('keras version:', keras.__version__)
print('tensorflow version:', tensorflow.__version__)   