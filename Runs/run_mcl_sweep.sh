#!/usr/bin/env bash
# MCL inflation sweep on in-degree capped graph
# Usage: ./run_mcl_sweep.sh [GRAPH_ABC] [THREADS]
#   GRAPH_ABC: path to ABC-format graph (default: ../graph_capped100_d10.tsv)
#   THREADS:   expansion threads (default: 16)

set -euo pipefail

GRAPH="${1:-../graph_capped100_d10.tsv}"
THREADS="${2:-16}"
INFLATIONS=(1.4 2.0 3.0 4.0 5.0 6.0)

echo "=== MCL Inflation Sweep ==="
echo "Graph:   $GRAPH"
echo "Threads: $THREADS"
echo "Inflations: ${INFLATIONS[*]}"
echo ""

# Step 1: Convert ABC to binary format (much faster for repeated runs)
TAB_FILE="graph.tab"
MCI_FILE="graph.mci"

if [[ -f "$MCI_FILE" && -f "$TAB_FILE" ]]; then
    echo "Binary graph already exists, skipping conversion."
else
    echo "Converting ABC to binary format..."
    time mcxload -abc "$GRAPH" \
        --stream-mirror \
        -write-tab "$TAB_FILE" \
        -o "$MCI_FILE"
    echo ""
fi

# Step 2: Run MCL at each inflation value
for I in "${INFLATIONS[@]}"; do
    OUT="mcl_I${I}.clusters"
    echo "================================================================"
    echo "Running MCL with inflation = $I"
    echo "================================================================"
    time mcl "$MCI_FILE" -I "$I" -te "$THREADS" -o "$OUT" -use-tab "$TAB_FILE"
    echo ""
done

# Step 3: Parse results
echo ""
echo "================================================================"
echo "Parsing results..."
echo "================================================================"
echo ""

# Header for summary
printf "%-8s  %10s  %10s  %8s  %10s  %10s  %10s  %10s\n" \
    "I" "clusters" "singletons" "pct_sing" "non_sing" "largest" "clustered" "pct_clust"
printf "%s\n" "$(printf '=%.0s' {1..96})"

for I in "${INFLATIONS[@]}"; do
    OUT="mcl_I${I}.clusters"
    if [[ ! -f "$OUT" ]]; then
        echo "  $OUT not found, skipping"
        continue
    fi

    # Each line is a cluster (tab-separated labels)
    total_clusters=$(wc -l < "$OUT")
    singletons=$(awk -F'\t' 'NF==1{n++} END{print n+0}' "$OUT")
    non_singletons=$((total_clusters - singletons))
    total_seqs=$(awk -F'\t' '{s+=NF} END{print s}' "$OUT")
    clustered=$((total_seqs - singletons))
    pct_sing=$(awk "BEGIN{printf \"%.1f\", 100.0 * $singletons / $total_seqs}")
    pct_clust=$(awk "BEGIN{printf \"%.1f\", 100.0 * $clustered / $total_seqs}")
    largest=$(awk -F'\t' '{if(NF>m)m=NF} END{print m}' "$OUT")

    printf "%-8s  %10d  %10d  %7s%%  %10d  %10d  %10d  %9s%%\n" \
        "$I" "$total_clusters" "$singletons" "$pct_sing" "$non_singletons" \
        "$largest" "$clustered" "$pct_clust"
done

echo ""

# Detailed top-20 for each inflation
for I in "${INFLATIONS[@]}"; do
    OUT="mcl_I${I}.clusters"
    [[ ! -f "$OUT" ]] && continue

    echo "--- I = $I : Top 20 communities ---"
    awk -F'\t' '{print NF}' "$OUT" | sort -rn | head -20 | \
        awk '{printf "  %3d. %d sequences\n", NR, $1}' || true
    echo ""
done

echo "=== Sweep complete ==="
