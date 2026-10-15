# SynthStrip skull stripping (Docker)

Brain masks come from FreeSurfer's [SynthStrip](https://synthstrip.io) run in Docker, so no FreeSurfer
installation is needed. `run_synthstrip.sh` writes `<name>_brain.nii.gz` and `<name>_mask.nii.gz` for each input.

```bash
docker pull freesurfer/synthstrip:1.8                       # CPU image
docker build -t synthstrip-gpu:1.8 - < Dockerfile.gpu        # optional GPU image (adds CUDA PyTorch)
# RTX 50-series GPUs need a newer PyTorch:
docker build --build-arg TORCH_VERSION=2.7.1 --build-arg CUDA_TAG=cu128 -t synthstrip-gpu:1.8 - < Dockerfile.gpu

./run_synthstrip.sh -in image.nii.gz -out outdir             # GPU image (default)
./run_synthstrip.sh --cpu -in image.nii.gz -out outdir       # CPU image
./run_synthstrip.sh -in image.nii.gz -out outdir --no-csf    # extra flags go to mri_synthstrip
```

GPU use needs the NVIDIA Container Toolkit; the script uses `--runtime=nvidia`. If your Docker setup expects
`--gpus all` instead, edit `DOCKER_GPU` in `run_synthstrip.sh`. `scripts/skullstrip_dataset.sh` runs this
script for every subject of a dataset split.

Citation: Hoopes et al., "SynthStrip: Skull-Stripping for Any Brain Image", NeuroImage 2022.
