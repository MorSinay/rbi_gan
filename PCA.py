import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from pathlib import Path
home = str(Path.home())

root_dir = os.path.join(home,'Desktop/temp/moraband/')

pi_file = os.path.join(root_dir, "pi.npy")
pi_explore_file = os.path.join(root_dir, "pi_explore.npy")
pi = np.load(pi_file)
pi_explore = np.load(pi_explore_file)

# pi = pi[:1]
# pi_explore = pi_explore[:1]

one_hot = np.zeros((10,10))
one_hot[np.arange(10), np.arange(10)] = 1

uniform = np.ones((1,10))/10

pca = PCA(n_components=2)
np.seterr(divide='ignore', invalid='ignore')
X_train = pca.fit_transform(pi)
X_test = pca.transform(pi_explore)

one_hot_pca = pca.transform(one_hot)
uniform_pca = pca.transform(uniform)


plt.plot(X_train[:,0], X_train[:,1], 'rx', label='pi')
plt.plot(X_test[:,0], X_test[:,1], 'y+', label='pi_explore')

colors = ['ko','mo','co','bo','k*','m*','c*','b*','g*','kx']
for i in range(10):
    plt.plot(one_hot_pca[i,0], one_hot_pca[i,1], colors[i], label='one_hot_{}_pca'.format(i))

plt.plot(uniform_pca[0,0], uniform_pca[0,1], 'go', label='uniform')

plt.legend(loc='best', shadow=True, fontsize='x-small')

path = os.path.join(root_dir, 'pca.png')
plt.show()
plt.savefig(path, bbox_inches='tight')
plt.close()


# grad_file = os.path.join(root_dir, "grad.npy")
# grad = np.load(grad_file)
# grad = np.sum(np.abs(grad), axis=1)
# plt.plot(np.arange(grad.shape[0]), grad, 'ko')
# plt.show()
# plt.close()
