import os
import numpy as np
import cv2
from test_final import get_embeds
from numpy import savez_compressed

arr = os.listdir('/home/siddhartha/Siddhartha/Reveal-Media/Reveal Videos/redaction/train')
list = []
for x in arr:
    folder = os.path.join('/home/siddhartha/Siddhartha/Reveal-Media/Reveal Videos/redaction/train/',x)
    for filename in os.listdir(folder):
        img_raw = cv2.imread(os.path.join(folder,filename))
        if img_raw is not None:
            imgpth = os.path.join(folder,filename)
        embeds = get_embeds(img_raw).numpy()
        embed = np.append(embeds,int(x))
        print(embed.shape)
        list.append(embed)
dataembed = np.asarray(list)
savez_compressed('dataembed.npz', dataembed)

