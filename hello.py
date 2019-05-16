import matplotlib.pyplot as plt
from memory_fmnist import Singleton_Mem
import numpy as np
from tic_toc import tic,toc
from environment import train_label_0_model_fmnist

train_label_0_model_fmnist(250, 128, 4)
