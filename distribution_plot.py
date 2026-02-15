import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import scienceplots

plt.style.use(['science','no-latex'])

datadf = pd.read_csv('distribution_plot.csv', delimiter=',')
datadf = datadf.to_numpy()

plt.hist(datadf, bins=np.arange(min(datadf), max(datadf)+0.48, step=0.48))
plt.title('Distribution plot of collated piezoelectric coefficients')
plt.xlabel("d33")
plt.ylabel("Count")

plt.show()
