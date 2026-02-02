# Clustering Analysis Notes

## 2025-11-28 19:50

### Initial UMAP/HDBSCAN Results
- Ran UMAP + HDBSCAN on VAE latent space (256 dimensions)
- HDBSCAN found only 2 clusters - not very informative
- However, UMAP visualization shows two large clouds with internal structure

### Distance Distribution Analysis
- Sampled 1,000 embeddings, computed pairwise distances
- **Euclidean distances**: unimodal distribution
- **Cosine distances**: clearly bimodal distribution

### Hypothesis: Eukaryotes vs Prokaryotes
The bimodal cosine distance distribution and two-cloud UMAP structure suggest a fundamental biological division. Eukaryotes vs Prokaryotes is a strong candidate because:
1. Distinct k-mer signatures (codon usage bias, GC content, repetitive elements)
2. The split is a primary division, not subtle clustering

### Next Steps to Validate
- Extract representative sequences from each cloud and BLAST them
- Check if GC content differs systematically between clouds
- Use k-means (k=2) to label points and examine sequences from each group

## 2025-11-28 20:05

### Validation: Latent Space Reflects Taxonomy
- Found the embedding closest to the first sequence in latent space
- BLAST results confirm they are taxonomically close
- **Conclusion**: VAE latent space is capturing meaningful biological relationships
- Sequences close in cosine distance are close taxonomically

---

## 2025-12-03 ~19:00: Distance distribution analysis (VAE, 384-dim)

### Observation

Re-ran distance analysis with new VAE embeddings (384-dim, beta=0.05):
- **Euclidean distance**: Unimodal distribution
- **Cosine distance**: Bimodal distribution

Same pattern as the original 256-dim VAE, confirming the structure is preserved.

### Interpretation

**Euclidean distance** measures absolute distance in the 384-dimensional space. In high dimensions, distances tend to concentrate around a mean value (the "curse of dimensionality") - most points end up roughly the same distance apart, giving a unimodal distribution.

**Cosine distance** measures the angle between vectors, ignoring magnitude. This is sensitive to the *direction* of vectors in the latent space. The bimodal distribution suggests there are two dominant "directions" or clusters in the latent space - possibly:
- Chromosomal DNA vs. mobile elements (plasmids/viruses)
- Bacteria vs. archaea
- High GC vs. low GC organisms
- Or some other fundamental compositional split in the data

### Significance

This is a good sign - it means the VAE is learning meaningful structure that separates major groups. The bimodality in cosine distance is why ChromaDB is configured with `'hnsw:space': 'cosine'` - it should be better at distinguishing these groups for retrieval.
