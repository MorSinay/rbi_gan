import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from pathlib import Path
home = str(Path.home())

root_dir = os.path.join(home,'Desktop/temp/yoda/')

pi_file = os.path.join(root_dir, "pi.npy")
pi_explore_file = os.path.join(root_dir, "pi_explore.npy")
pi = np.load(pi_file)
pi_explore = np.load(pi_explore_file)

pi = pi[1000:]
pi_explore = pi_explore[1000:]

length = pi.shape[0]
pca = PCA(n_components=2)
np.seterr(divide='ignore', invalid='ignore')
X_train = pca.fit_transform(pi)
X_test = pca.transform(pi_explore)
plt.plot(X_train[:,0], X_train[:,1], 'ro', label='pi')
plt.plot(X_test[:,0], X_test[:,1], 'bo', label='pi_explore')
plt.legend(loc='upper center', shadow=True, fontsize='x-large')

path = os.path.join(root_dir, 'pca.png')
plt.show()
plt.savefig(path, bbox_inches='tight')
#plt.close()
