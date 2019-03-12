#%%
import numpy as np
import tensorflow as tf

#%%
M = np.load("masses.npy")
V = np.load("initial_velocities.npy")
X = np.load("initial_positions.npy")
G = 6.67e5
timestep = 1e-4
end_time = 1.0
n = M.shape[0]

#%%
masses = tf.placeholder(float, shape=(n, 1))
vels = tf.placeholder(float, shape=(n, 2))
pos = tf.placeholder(float, shape=(n, 2))

pos_mat = tf.reshape(tf.tile(pos, (n, 1)), shape=(n, n, 2))
pos_rel = tf.subtract(pos_mat, tf.transpose(pos_mat, (1, 0, 2)))
pos_div = (pos_rel[:, :, 0] - pos_rel[:, :, 1]) ** 3
npos_0 = tf.div_no_nan(pos_rel[:, :, 0], pos_div)
npos_1 = tf.div_no_nan(pos_rel[:, :, 1], pos_div)
acc_0 = -G * tf.linalg.matmul(npos_0, masses)
acc_1 = -G * tf.linalg.matmul(npos_1, masses)
acc = tf.reshape(
    tf.stack((tf.reshape(acc_0, (n, 1)), tf.reshape(acc_1, (n, 1))), axis=1), (n, 2)
)
new_vels = tf.add(vels, timestep * acc)
new_pos = tf.add(pos, tf.add(new_vels * timestep, 0.5 * (timestep ** 2) * acc))

#%%
pos_list, vel_list = list(), list()
newV, newX = V, X
time = 0.0
with tf.Session() as sess:
    while time < end_time:
        time += timestep
        newV, newX = sess.run([new_vels, new_pos], {masses: M, vels: newV, pos: newX})

#%%
np.save("positions.npy", newX)
np.save("velocities.npy", newV)
