# CAD-VAE for Crickets dataset

PyTorch adaptation of CAD-VAE for cricket SNP data.

## Install

From the repository root, create and activate the Conda environment.

```bash
conda env create -n vae -f environment.yml
conda activate vae
```

## Run

Set the dataset paths, code paths, and Conda setup in the Slurm scripts for your cluster. Create the log directory specified in `#SBATCH --output` and `#SBATCH --error` before submitting.

### Train all LD blocks

```bash
sbatch run_cadvae_all_blocks.sh
```

This runs five-fold cross-validation for each LD block across scaffolds 1–14.

### Run ablation studies

Select a training job and submit its plotting job after the matching runs finish.

| Experiment | Train | Plot |
| --- | --- | --- |
| CMI and classification weights | `sbatch run_cricket_cmi.sh` | `sbatch run_plot_cricket_cmi.sh` |
| LRI weight | `sbatch run_cricket_lri.sh` | `sbatch run_plot_cricket_lri.sh` |
| TC weight | `sbatch run_cricket_tc.sh` | `sbatch run_plot_cricket_tc.sh` |
