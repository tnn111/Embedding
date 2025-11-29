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
