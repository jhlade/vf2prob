#!/usr/bin/env bash
# Ground-truth recovery vs. the fixed node weights (node_w0, node_w_label) on a grid,
# edge weights held at their defaults, via recovery --fixed-weights at each grid point.
# Writes results/sweep_weights.csv: w0,wl,flip,method,accuracy,n.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REC="$HERE/vf2prob-sys-cpp/build/recovery"
OUT="$HERE/results/sweep_weights.csv"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
INSTANCES="${INSTANCES:-100}"
FLIPS="${FLIPS:-0.05,0.15,0.3}"
W0S="${W0S:--5 -3 -1}"          # node_w0 grid
WLS="${WLS:-3 6 9}"             # node_w_label grid
EDGE="-1 3 2"                   # edge_w0 edge_w_label edge_w_prob (defaults)

echo "w0,wl,flip,method,accuracy,n" > "$OUT"
for w0 in $W0S; do
  for wl in $WLS; do
    echo "$w0 $wl $EDGE" > "$TMP/w.txt"
    "$REC" --instances "$INSTANCES" --flips "$FLIPS" \
      --methods vf2prob-astar-assign --learned 0 \
      --fixed-weights "$TMP/w.txt" --out "$TMP/rec.csv" >/dev/null
    # rec.csv header: flip,method,accuracy,n
    tail -n +2 "$TMP/rec.csv" | while IFS=, read -r flip method acc n; do
      echo "$w0,$wl,$flip,$method,$acc,$n" >> "$OUT"
    done
    echo "  w0=$w0 wl=$wl done"
  done
done
echo "wrote $OUT"
