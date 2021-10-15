from ISR.models import RDN
from local import data_folder
from torchvision.datasets import  STL10
import numpy as np
import matplotlib.pyplot as plt



rdn = RDN(weights='psnr-small')

stl10 = STL10(data_folder, 'train', download=False)

for img, _ in stl10:

    lr_img = np.array(img)
    sr_img = rdn.predict(lr_img)

    fig, ax = plt.subplots(1, 2, figsize=(20,10), sharex='all', sharey='all')

    print(lr_img.shape, sr_img.shape)

    ax[0].imshow(img.resize(sr_img.shape[:2]))
    ax[1].imshow(sr_img)

    plt.show()