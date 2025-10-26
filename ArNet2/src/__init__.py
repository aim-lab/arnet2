import os
import numpy as np
import random
import utils.consts as cts

# Random
np.random.seed(cts.SEED)
random.seed(cts.SEED)

# Declare GPU parameters
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the GPU you wish to use here
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
