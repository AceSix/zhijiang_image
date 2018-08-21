import os
import socket
hostname=socket.gethostname()

# path
DATA_PATH="./DatasetA_train_20180813"
TEST_PATH="./DatasetA_test_20180813"
LOG_PATH="./logs"

#model
IMAGE_SIZE=64
REGUL=0.0
attributes=[]
classes=[]
attribute_txt=os.path.join(DATA_PATH,"attributes_per_class.txt")
with open(attribute_txt,'r') as f:
    for x in f.readlines():
        classes.append(x.strip().split()[0])
        attributes.append([float(y) for y in x.strip().split()[1:]])
ATTRIBUTES=attributes
CLASSES=classes

# solver parameter
BATCH_SIZE=512
MAX_ITER=10000
SUMMARY_ITER=100
LEARNING_RATE = 1e-4

if hostname=="yangzhan-PC":
    BATCH_SIZE=128
    MAX_ITER=40000
elif hostname=="yzmac":
    BATCH_SIZE=8
    MAX_ITER=200