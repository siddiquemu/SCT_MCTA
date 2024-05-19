import pandas as pd
import pdb
import matplotlib.pyplot as plt
from adtk.detector import LevelShiftAD
s = pd.read_excel('/home/siddique/Downloads/CPU_ransomware.xlsx')
#s = validate_series(s)

pdb.set_trace()
level_shift_ad = LevelShiftAD(c=6.0, side='both', window=5)
anomalies = level_shift_ad.fit_detect(s)
#plt.plot(s, anomaly=anomalies, anomaly_color='red');