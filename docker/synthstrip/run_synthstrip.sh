#!/usr/bin/env bash
# Skull-strip one or more NIfTI/MGZ images with the SynthStrip Docker image (GPU by default).
# Results go to the output directory: <name>_brain.nii.gz and <name>_mask.nii.gz.
# Usage: ./run_synthstrip.sh -in path/to/image.nii.gz -out OUTDIR
#        ./run_synthstrip.sh -in img1.nii.gz -in img2.nii.gz -out OUTDIR   (several images)
#        ./run_synthstrip.sh -in image.nii.gz                    (asks for the output directory)
#        ./run_synthstrip.sh --cpu -in image.nii.gz -out OUTDIR  (no GPU, uses freesurfer/synthstrip:1.8)
#        ./run_synthstrip.sh -in image.nii.gz -out OUTDIR --no-csf  (other options are passed to mri_synthstrip)
set -euo pipefail

IMAGE="synthstrip-gpu:1.8"
DOCKER_GPU=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all); SS_GPU=(-g)
files=(); extra=(); OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cpu) IMAGE="freesurfer/synthstrip:1.8"; DOCKER_GPU=(); SS_GPU=() ;;
    --gpu) ;;  # default, accepted for compatibility
    -in|--in|-i) [ $# -ge 2 ] || { echo "$1 needs an image path"; exit 1; }; files+=("$2"); shift ;;
    -out|--out|-o) [ $# -ge 2 ] || { echo "$1 needs a directory"; exit 1; }; OUT="$2"; shift ;;
    *.nii|*.nii.gz|*.mgz) files+=("$1") ;;
    *) extra+=("$1") ;;
  esac
  shift
done
[ ${#files[@]} -eq 0 ] && { echo "Usage: $0 [--cpu] -in image.nii.gz [-in more.nii.gz ...] -out OUTDIR [mri_synthstrip options]"; exit 1; }

if [ -z "$OUT" ]; then
  read -r -e -p "Output directory: " OUT
  [ -n "$OUT" ] || { echo "No output directory given."; exit 1; }
fi
OUT="${OUT/#\~/$HOME}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

command -v docker >/dev/null || { echo "Docker not found."; exit 1; }
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  if [ ${#SS_GPU[@]} -gt 0 ]; then
    echo "Image $IMAGE not found. Build it first:  docker build -t $IMAGE - < Dockerfile.gpu"; exit 1
  fi
  docker pull "$IMAGE"
fi

for f in "${files[@]}"; do
  [ -f "$f" ] || { echo "File not found: $f"; exit 1; }
  dir="$(cd "$(dirname "$f")" && pwd)"
  name="$(basename "$f")"; base="${name%.nii.gz}"; base="${base%.nii}"; base="${base%.mgz}"
  echo ">> $f"
  docker run --rm "${DOCKER_GPU[@]}" -u "$(id -u):$(id -g)" \
    -v "$dir":/data/input:ro -v "$OUT":/data/output \
    "$IMAGE" \
    -i "/data/input/$name" \
    -o "/data/output/${base}_brain.nii.gz" \
    -m "/data/output/${base}_mask.nii.gz" "${SS_GPU[@]}" "${extra[@]}"
done
echo "Done. Results are in $OUT"
