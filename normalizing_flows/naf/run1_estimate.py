from __future__ import print_function
from __future__ import division

"""
## imports
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils import *

"""
## ini
"""
batch_size = 64
nb_its = 200

"""
## data
"""
N1 = 200
mu1 = np.array([-5, -5])
Sigma1 = np.array([[2, 1], [1, 2]])

N2 = 300
mu2 = np.array([+4, +5])
Sigma2 = np.array([[2, -1], [-1, 2]])

"""
## Data generation
"""
X1 = np.random.multivariate_normal(mu1, Sigma1, N1)
X2 = np.random.multivariate_normal(mu2, Sigma2, N2)
X = np.concatenate((X1, X2), axis=0)

"""
# Build FLow
"""
from net1 import Net
net = Net(2)
net.train(X, batch_size=64, nb_its=200)

"""
## view
"""
x1 = np.linspace(-20, 20, 100)
x2 = np.linspace(-20, 20, 100)
dx1 = x1[1] - x1[0]
dx2 = x2[1] - x2[0]
xg1, xg2 = np.meshgrid(x1, x2)
x = np.array([xg1.flatten(), xg2.flatten()]).T

log_p = net.log_p(x).data.numpy()

p = np.exp(log_p)
print(np.sum(p) * dx1 * dx2)

plt.figure()
plt.imshow(log_p.reshape(100, 100), extent=[-20, 20, -20, 20], origin='lower', cmap='jet')
plt.contour(log_p.reshape(100, 100), 30, extent=[-20, 20, -20, 20], origin='lower', colors='white')
plt.savefig('out/run1_fig1.png', dpi=300)

plt.figure()
plt.imshow(p.reshape(100, 100), extent=[-20, 20, -20, 20], origin='lower', cmap='jet')
plt.savefig('out/run1_fig2.png', dpi=300)

plt.figure()
plt.imshow(p.reshape(100, 100), extent=[-20, 20, -20, 20], origin='lower', cmap='jet')
plt.plot(X[:,0],X[:,1],'wo',markersize=1)
plt.savefig('out/run1_fig3.png', dpi=300)

plt.figure()
plt.imshow(log_p.reshape(100, 100), extent=[-20, 20, -20, 20], origin='lower', cmap='jet')
plt.contour(log_p.reshape(100, 100), 30, extent=[-20, 20, -20, 20], origin='lower', colors='white')
plt.plot(X[:,0],X[:,1],'wo',markersize=2)
plt.savefig('out/run1_fig4.png', dpi=300)

"""
## data transformation
"""

z1 = net.get_z(X1).data.numpy()
z2 = net.get_z(X2).data.numpy()

plt.figure()
plt.plot(z1[:,0],z1[:,1],'o',markersize=2)
plt.plot(z2[:,0],z2[:,1],'o',markersize=2)
plt.savefig('out/run1_fig5.png', dpi=300)