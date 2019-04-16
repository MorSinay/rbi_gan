import matplotlib.pyplot as plt
from memory_fmnist import Singleton_Mem
from Dummy_Gen import DummyGen
import numpy as np
from tic_toc import tic,toc
from environment import train_primarily_model

train_primarily_model(250, 128, 5)
