import pandas as pd
import matplotlib.pyplot as plt

import scienceplots
import seaborn as sns

plt.style.use(['science','no-latex'])

temp_d33 = pd.read_csv('/Users/admin/projects/AlN/ML MODELS/temperature.csv')
sns.set_style("whitegrid")
sns.stripplot(x = 'Temperature', y = 'd33', marker = '^', size = 6, alpha = 0.6, data = temp_d33)
sns.boxplot(x = 'Temperature', y = 'd33', data = temp_d33)
plt.xlabel('Temperature / degrees Celsius')
plt.ylabel('d33, f (pC/N)')
plt.show() 