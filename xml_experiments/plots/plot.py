import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import *


def plot_data(title, d, remove_reg=False):

    if remove_reg:
        d['type'].pop(13)
        d['type'].pop(9)
        d['type'].pop(5)
        d['type'].pop(1)
        d['mod'].pop(13)
        d['mod'].pop(9)
        d['mod'].pop(5)
        d['mod'].pop(1)
        d['p1'].pop(13)
        d['p1'].pop(9)
        d['p1'].pop(5)
        d['p1'].pop(1)

    df = pd.DataFrame(data=d)
    print(df)

    g = sns.catplot(x="type", y="p1", col="mod", data=df, ci=None, aspect=.3, kind="bar")
    g.set_axis_labels("", "precision@1") \
        .set_xticklabels(["HSM", "PLT"]) \
        .set_titles("{col_name}") \
        .set(ylim=(floor(min(d["p1"]) * 10 ) / 10 - 0.05, ceil(max(d["p1"]) * 10 ) / 10 )) \
        .despine(left=True)

    plt.tight_layout()
    #plt.show()
    #g.savefig(title + ".png")
    g.savefig(title + ".pdf")

# Eurlex
d = {
    'type': ['HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT'],
    'mod': ['huffman', 'huffman + reg.', 'huffman\n+ tf-idf', 'huffman\n+ tf-idf + reg.', 'top-down\nclustering', 'top-down\nclustering + reg.', 't-d clustering\n+ tf-idf', 't-d clustering\n+ tf-idf + reg.',
            'huffman', 'huffman + reg.', 'huffman\n+ tf-idf', 'huffman\n+ tf-idf + reg.', 'top-down\nclustering', 'top-down\nclustering + reg.', 't-d clustering\n+ tf-idf', 't-d clustering\n+ tf-idf + reg.'],
    'p1': [0.58717, 0.55325, 0.69734, 0.70181, 0.59032, 0.56166, 0.69235, 0.70865, 0.63397, 0.62793, 0.71207, 0.74599, 0.68314, 0.65054, 0.75046, 0.77676]
}
plot_data("Eurlex", d)
plot_data("Eurlex_new_no_reg", d, remove_reg=True)

# Amazon 670k
d = {
    'type': ['HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT'],
    'mod': ['huffman', 'huffman + reg.', 'huffman\n+ tf-idf', 'huffman\n+ tf-idf + reg.', 'top-down\nclustering', 'top-down\nclustering + reg.', 't-d clustering\n+ tf-idf', 't-d clustering\n+ tf-idf + reg.',
            'huffman', 'huffman + reg.', 'huffman\n+ tf-idf', 'huffman\n+ tf-idf + reg.', 'top-down\nclustering', 'top-down\nclustering + reg.', 't-d clustering\n+ tf-idf', 't-d clustering\n+ tf-idf + reg.'],
    'p1': [0.20193, 0.17136, 0.29393, 0.2994, 0.23977, 0.25664, 0.3062, 0.36693, 0.23194, 0.21642, 0.2976, 0.32111, 0.28538, 0.2895, 0.36239, 0.39415]
}
plot_data("Amazon-670K", d)
plot_data("Amazon-670K_new_no_reg", d, remove_reg=True)

# Amazon 670k_new
d = {
    'type': ['HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT'],
    'mod': ['huffman', 'huffman + reg.', 'huffman\n+ tf-idf', 'huffman\n+ tf-idf + reg.', 'top-down\nclustering', 'top-down\nclustering + reg.', 't-d clustering\n+ tf-idf', 't-d clustering\n+ tf-idf + reg.',
            'huffman', 'huffman + reg.', 'huffman\n+ tf-idf', 'huffman\n+ tf-idf + reg.', 'top-down\nclustering', 'top-down\nclustering + reg.', 't-d clustering\n+ tf-idf', 't-d clustering\n+ tf-idf + reg.'],
    'p1': [0.24193, 0.20136, 0.29393, 0.2994, 0.23977, 0.25664, 0.3062, 0.36693, 0.23194, 0.21642, 0.2976, 0.32111, 0.28538, 0.2895, 0.36239, 0.39415]
}
plot_data("Amazon-670K_new", d)
plot_data("Amazon-670K_new_no_reg", d, remove_reg=True)


# WikiLSHTC
d = {
    'type': ['HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT'],
    'mod': ['huffman', 'huffman + reg.', 'huffman\n+ tf-idf', 'huffman\n+ tf-idf + reg.', 'top-down\nclustering', 'top-down\nclustering + reg.', 't-d clustering\n+ tf-idf', 't-d clustering\n+ tf-idf + reg.',
            'huffman', 'huffman + reg.', 'huffman\n+ tf-idf', 'huffman\n+ tf-idf + reg.', 'top-down\nclustering', 'top-down\nclustering + reg.', 't-d clustering\n+ tf-idf', 't-d clustering\n+ tf-idf + reg.'],
    'p1': [0.18651, 0.41973, 0.2014, 0.43734, 0.23104, 0.54222, 0.24771, 0.54833, 0.25837, 0.24007, 0.24883, 0.24297, 0.34436, 0.55951, 0.36724, 0.58356]
}
plot_data("WikiLSHTC", d)
plot_data("WikiLSHTC_no_reg", d, remove_reg=True)


# WikiLSHTC_new
d = {
    'type': ['HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'HSM', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT', 'PLT'],
    'mod': ['huffman', 'huffman + reg.', 'huffman\n+ tf-idf', 'huffman\n+ tf-idf + reg.', 'top-down\nclustering', 'top-down\nclustering + reg.', 't-d clustering\n+ tf-idf', 't-d clustering\n+ tf-idf + reg.',
            'huffman', 'huffman + reg.', 'huffman\n+ tf-idf', 'huffman\n+ tf-idf + reg.', 'top-down\nclustering', 'top-down\nclustering + reg.', 't-d clustering\n+ tf-idf', 't-d clustering\n+ tf-idf + reg.'],
    'p1': [0.4113, 0.41973, 0.41274, 0.43734, 0.50524, 0.54222, 0.51771, 0.54833, 0.40717, 0.42464, 0.40825, 0.42643, 0.49326, 0.55951, 0.51724, 0.58356]
}
plot_data("WikiLSHTC_new", d)
plot_data("WikiLSHTC_new_no_reg", d, remove_reg=True)
