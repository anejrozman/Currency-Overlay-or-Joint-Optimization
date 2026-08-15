from matplotlib.pyplot import plot

import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("oos_returns.csv", index_col=0, parse_dates=True)["return"]
cumulative_returns = (1 + data).cumprod() - 1
ax = cumulative_returns.plot(figsize=(10, 6), title="Cumulative Returns")
plt.show()
