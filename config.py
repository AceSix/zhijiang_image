import os
import socket
hostname=socket.gethostname()
# path
DATA_PATH="./DatasetA_train_20180813"
LOG_PATH="./logs"

#model
IMAGE_SIZE=64
REGUL=0.0


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