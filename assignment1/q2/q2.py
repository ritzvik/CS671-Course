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
threshold_dist = 0.1

#%%
masses = tf.placeholder(float, shape=(n, 1))
vels = tf.placeholder(float, shape=(n, 2))
pos = tf.placeholder(float, shape=(n, 2))
pos_mat = tf.reshape(tf.tile(pos, (n, 1)), shape=(n, n, 2))
pos_rel = tf.subtract(pos_mat, tf.transpose(pos_mat, (1, 0, 2)))
pos_rel_sq = pos_rel ** 2
pos_div = (pos_rel_sq[:, :, 0] + pos_rel_sq[:, :, 1]) ** (3 / 2)
npos_0 = tf.div_no_nan(pos_rel[:, :, 0], pos_div)
npos_1 = tf.div_no_nan(pos_rel[:, :, 1], pos_div)
acc_0 = -G * tf.linalg.matmul(npos_0, masses)
acc_1 = -G * tf.linalg.matmul(npos_1, masses)
acc = tf.reshape(tf.stack((acc_0, acc_1), axis=1), (n, 2))
new_vels = tf.add(vels, timestep * acc)
new_pos = tf.add(pos, tf.add(vels * timestep, 0.5 * (timestep ** 2) * acc))

#%%
pos_list, vel_list = list(), list()
pd = 10000 * np.ones((n, n), float)
newV, newX = V, X
time = 0.0
count = 0
with tf.Session() as sess:
    writer = tf.summary.FileWriter("./graphs", sess.graph)
    np.fill_diagonal(pd, 10000)
    tmp = np.amin(pd)
    while tmp > threshold_dist:
        count += 1
        # print(tmp)
        time += timestep
        pos_list.append(newX)
        vel_list.append(newV)
        pd, newV, newX = sess.run(
            [pos_div, new_vels, new_pos], {masses: M, vels: newV, pos: newX}
        )
        pd = pd ** (1 / 3)
        np.fill_diagonal(pd, 10000)
        tmp = np.amin(pd)

#%%
np.save("positions.npy", newX)
np.save("velocities.npy", newV)
