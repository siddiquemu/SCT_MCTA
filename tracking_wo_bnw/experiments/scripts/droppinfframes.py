#read all frames
from PIL import Image
import os
import glob
files = glob.glob('/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/cam02exp2.mp4/*.png')
files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
#dropping frames at specific rates
new_fr=1
for i, file in enumerate(files):
    if (i+1)%2==0:
        print(file)
        img = Image.open(file)
        img.save('/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/15FPS/cam02exp2.mp4' + '/{:06d}.png'.format(new_fr))
        new_fr += 1
