import os
import numpy as np
import matplotlib.pyplot as plt

res_dir = None
for d in os.listdir('.'):
    if os.path.isdir(d) and 'result' in d and 'old' not in d:
        res_dir = d
        break

res_list = []
for d in os.listdir(res_dir):
    map_file = os.path.join(res_dir, os.path.join(d, 'map.txt'))
    with open(map_file, 'r') as f:
        lines = f.readlines()
        i2t = float(lines[2][24:29])
        t2i = float(lines[2][42:47])
        t2t = float(lines[3][24:29])
        i2i = float(lines[3][42:47])
        res_list.append([i2t, t2i])

inds = np.array(list(range(len(res_list))))
res = np.array(res_list).transpose()
plt.plot(inds, res[0, :])
plt.plot(inds, res[1, :])
plt.plot(inds, np.mean(res, axis=0))
plt.show()
print(np.mean(res, axis=0).max())