import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import *


def plot_data(title, d):

    df = pd.DataFrame(data=d)
    print(df)

    g = sns.catplot(x="type", y="p1", col="mod", data=df, ci=None, aspect=.3, kind="bar")
    g.set_axis_labels("", "precision@1") \
        .set_titles("{col_name}") \
        .set(ylim=(floor(min(d["p1"]) * 10 ) / 10 - 0.05, ceil(max(d["p1"]) * 10 ) / 10 )) \
        .despine(left=True)

    plt.tight_layout()
    #plt.show()
    #g.savefig(title + ".png")
    g.savefig(title + ".pdf")

# Eurlex
d = {
    'type': ['HSM', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT'],
    'mod': ['FT\n(HSM huffman)', 'PLT huffman', 'PLT complete', 'PLT top-down\nclustering', 'PLT top-down\nclustering + reg.', 'PLT t-d clustering\n+ tf-idf', 'PLT t-d clustering\n+ tf-idf + reg.'],
    'p1': [0.58717, 0.63397, 0.6439, 0.68314, 0.65054, 0.75046, 0.77676]
}
plot_data("Eurlex_no_hsm", d)

# Amazon 670k_new
d = {
    'type': ['HSM', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT'],
    'mod': ['FT\n(HSM huffman)', 'PLT huffman', 'PLT complete', 'PLT top-down\nclustering', 'PLT top-down\nclustering + reg.', 'PLT t-d clustering\n+ tf-idf', 'PLT t-d clustering\n+ tf-idf + reg.'],
    'p1': [0.24193, 0.26136,0.2674, 0.28538, 0.2895, 0.36239, 0.39415]
}
plot_data("Amazon-670K_no_hsm", d)

# WikiLSHTC_new
d = {
    'type': ['HSM', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT'],
    'mod': ['FT\n(HSM huffman)', 'PLT huffman', 'PLT complete', 'PLT top-down\nclustering', 'PLT top-down\nclustering + reg.', 'PLT t-d clustering\n+ tf-idf', 'PLT t-d clustering\n+ tf-idf + reg.'],
    'p1': [0.4113, 0.42973, 0.43734, 0.49326, 0.55951, 0.51724, 0.59156]
}
plot_data("WikiLSHTC_no_hsm", d)
