#!/usr/bin/env bash
# Run external structural solvers (Glasgow, PathLAD+) on the SNAP instances exported
# by vf2prob_run --export-lad. Glasgow reads vertex-labelled LAD; PathLAD+ reads
# unlabelled LAD (lad_strip_labels.py). Writes results/ext_{glasgow,pathlad}_*.csv.
#
# Env overrides:
#   GLASGOW   path to glasgow_subgraph_solver (skipped if unset)
#   PATHLAD   path to PathLAD+ 'main' binary
#   TIMEOUT   per-instance CPU seconds
#   SIZES     space-separated sizes (default "S M L")
#   GRAPHS    space-separated "name:graphml:queryprefix"
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$HERE/vf2prob-sys-cpp/build/vf2prob_run"
DATA="$HERE/data/snap"
QDIR="$HERE/queries/snap"
OUT="$HERE/results"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
TIMEOUT="${TIMEOUT:-10}"
SIZES="${SIZES:-S M L}"
PATHLAD="${PATHLAD:-}"   # path to PathLAD+ 'main' (skipped if unset)
GLASGOW="${GLASGOW:-}"   # path to glasgow_subgraph_solver (skipped if unset)
# name:graphml:queryprefix
GRAPHS_DEFAULT="facebook:$DATA/facebook_combined.graphml:facebook \
ca-hepth:$DATA/ca_hepth.graphml:hep-th \
email-enron:$DATA/email_enron.graphml:enron"
GRAPHS="${GRAPHS:-$GRAPHS_DEFAULT}"

ulimit -s 102400 2>/dev/null || true
mkdir -p "$OUT"

pathlad_run() { # pattern target -> "<solutions> <seconds> <timeout?>"
  local p="$1" t="$2" extra="${3:-}"
  local o; o="$("$PATHLAD" -p "$p" -t "$t" -s "$TIMEOUT" $extra 2>/dev/null || true)"
  # "Run completed: N solutions; ...; X.XXXXXX seconds" / "CPU time exceeded"
  local sols secs to=0
  sols="$(printf '%s' "$o" | grep -oE '[0-9]+ solutions' | grep -oE '^[0-9]+' || echo NA)"
  secs="$(printf '%s' "$o" | grep -oE '[0-9.]+ seconds' | grep -oE '^[0-9.]+' || echo NA)"
  printf '%s' "$o" | grep -qi "time exceeded" && to=1
  echo "$sols $secs $to"
}

echo "PathLAD+ : ${PATHLAD:-<unset, skipped>}"
echo "Glasgow  : ${GLASGOW:-<unset, skipped>}"

for entry in $GRAPHS; do
  name="${entry%%:*}"; rest="${entry#*:}"; gml="${rest%%:*}"; qpref="${rest##*:}"
  [ -f "$gml" ] || { echo "skip $name (no $gml)"; continue; }
  for s in $SIZES; do
    qj="$QDIR/${qpref}_${s}.json"; [ -f "$qj" ] || { echo "skip $name $s (no $qj)"; continue; }
    ex="$TMP/${name}_${s}"; mkdir -p "$ex"
    "$RUN" --dataset graphml --data-path "$gml" --queries-path "$qj" --export-lad "$ex" >/dev/null
    # --- PathLAD+ (unlabelled) ---
    if [ -x "$PATHLAD" ]; then
      pout="$OUT/ext_pathlad_${name}_${s}.csv"
      echo "graph,size,query,mode,solutions,seconds,timeout" > "$pout"
      python3 "$HERE/scripts/lad_strip_labels.py" "$ex/target.lad" "$ex/target.plain.lad"
      for q in "$ex"/q*.lad; do
        case "$q" in *.plain.lad) continue;; esac
        qid="$(basename "$q" .lad | tr -dc 0-9)"
        python3 "$HERE/scripts/lad_strip_labels.py" "$q" "$ex/qp.lad"
        read d ds dto < <(pathlad_run "$ex/qp.lad" "$ex/target.plain.lad" "-f")  # decision
        read e es eto < <(pathlad_run "$ex/qp.lad" "$ex/target.plain.lad" "")    # enumerate
        echo "$name,$s,$qid,decision,$d,$ds,$dto" >> "$pout"
        echo "$name,$s,$qid,enumerate,$e,$es,$eto" >> "$pout"
      done
      echo "  wrote $pout"
    fi
    # --- Glasgow (vertex-labelled), only if a binary is given ---
    if [ -n "$GLASGOW" ] && command -v "$GLASGOW" >/dev/null 2>&1; then
      gout="$OUT/ext_glasgow_${name}_${s}.csv"
      echo "graph,size,query,mode,raw" > "$gout"
      for q in "$ex"/q*.lad; do
        case "$q" in *.plain.lad) continue;; esac
        qid="$(basename "$q" .lad | tr -dc 0-9)"
        r="$("$GLASGOW" --format lad --timeout "$TIMEOUT" "$q" "$ex/target.lad" 2>/dev/null | tr '\n' ' ' || true)"
        echo "$name,$s,$qid,decision,\"$r\"" >> "$gout"
      done
      echo "  wrote $gout"
    fi
  done
done
echo "done."
