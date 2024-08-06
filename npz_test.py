import numpy as np
import matplotlib.pyplot as plt
import config as conf
import os


npz_path=os.path.join(conf.train_folder, '869.npz')
np_data = np.load(npz_path, "rb", allow_pickle=True)

name1 = np_data['name1']
name2 = np_data['name2']


print(np.shape(name1), np.shape(name2))

for i in range(len(name1)):
    for frames in name1[i]:
        print(np.shape(frames))
        #subplot(r,c) provide the no. of rows and columns
        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        plt.imshow(frames)
        plt.show()
    print(f'label: {name2[i]}')
