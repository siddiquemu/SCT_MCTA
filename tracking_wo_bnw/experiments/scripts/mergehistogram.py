import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# set seed for reproducing
hists = pd.read_csv('/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/tracking_wo_bnw/output/tracktor/online_SCT/all_hist.csv',header=None,index_col=False)
hists = hists.values[1:-1]
merged_hist = np.sum(hists,axis=1)

plt.figure(figsize=(8,6))
bins = np.arange(0,2,2/50.)
plt.stem(bins[1::],merged_hist, label="combined hist")
plt.xlabel("Distance", size=14)
plt.ylabel("Count", size=14)
plt.title("Combined Histogram")
plt.legend(loc='upper right')

plt.show()
