#!/usr/bin/env bash
# Resume-friendly direct downloader for HF model files (sandbox-safe: no
# cache locks, no symlink staging, plain files only).
# Usage: bash curl_download.sh <repo> <dest_dir> <list_tsv>
#   list_tsv: "path<TAB>size" lines (from HF /api/models/<repo>/tree/main)
# Touches <dest_dir>/.download_done when every file matches its size.
set -u
REPO="$1"; DEST="$2"; LIST="$3"
fail_total=0
while IFS=$'\t' read -r p s; do
  [ -z "${p:-}" ] && continue
  p="${p%$'\r'}"; s="${s%$'\r'}"
  out="$DEST/$p"
  mkdir -p "$(dirname "$out")"
  cur=0; [ -f "$out" ] && cur=$(wc -c < "$out")
  if [ "$cur" = "$s" ]; then echo "skip $p"; continue; fi
  ok=0
  for i in 1 2 3 4 5 6 7 8; do
    curl -sS -L -C - --speed-limit 20480 --speed-time 45 -o "$out" \
      "https://huggingface.co/$REPO/resolve/main/$p" && { ok=1; break; }
    echo "retry$i $p rc=$?"; sleep 5
  done
  cur2=$(wc -c < "$out" 2>/dev/null || echo 0)
  if [ "$cur2" = "$s" ]; then echo "OK $p $s"; else echo "FAIL $p want=$s got=$cur2"; fail_total=$((fail_total+1)); fi
done < "$LIST"
if [ "$fail_total" -eq 0 ]; then
  date +%s > "$DEST/.download_done"
  echo "ALLDONE $REPO (marker written)"
else
  echo "ALLDONE $REPO with $fail_total FAILED FILES"
fi
