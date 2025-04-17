from imprints.data_collection import *
from imprints.data_cleaning import *

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-colorblind")
plt.figure(dpi=600)

PS_DATA_PATH = "../data/PS/data.csv"
E_DATA_PATH = "../data/E/data.csv"
F_DATA_PATH = "../data/F/data.csv"

df = pd.read_csv(PS_DATA_PATH, index_col=0)

# Figure 1

city_to_new_york = {"New York": 1}
df["new_york"] = df["city_group"].map(city_to_new_york).fillna(0).astype(int)

plot_df = (
    df[(df["year_min"] >= 1796) & (df["year_min"] <= 2010)]
    .groupby(["year_min", "new_york"])
    .size()
    .unstack()
    .fillna(0)
    .rolling(5)
    .mean()
)

plot_df = plot_df.div(plot_df.sum(axis=1), axis=0) * 100

plot_df.plot(kind="area", stacked=True)
plt.style.use("tableau-colorblind10")
plt.axhline(y=50, color="white", linestyle="--", linewidth=1)
plt.title("US Literature Imprint Location")
plt.xlabel("Year")
plt.ylabel("Percentage")
plt.legend(title="Published in New York")
