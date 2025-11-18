#!/usr/bin/env python

import  numpy as np
from    umap import UMAP
import  hdbscan


X_freq = np.loadtxt('./Data/kmer_frequencies_l5000_shuffled.txt', dtype=np.float32)

print(f"Loaded {X_freq.shape[0]} sequences")
print(f"Sum check: {X_freq[0].sum():.6f}")

reducer = UMAP(n_components = 32, metric = 'jensenshannon', n_neighbors = 30, min_dist = 0.0, verbose = True)

embedding = reducer.fit_transform(X_freq)
np.save('./Data/umap_32_30_0.npy', embedding)