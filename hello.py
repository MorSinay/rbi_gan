import matplotlib.pyplot as plt
from memory_fmnist import Singleton_Mem
import numpy as np
from tic_toc import tic,toc
from environment import train_primarily_model_mnist

train_primarily_model_mnist(250, 128, 5)
