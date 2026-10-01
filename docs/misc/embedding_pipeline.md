# Embedding Pipeline and Data Preparation

This document covers two CLI tools for preparing data for Stage 3 ProteoScribe finetuning.

## Embedding Pipeline

`biom3_embedding_pipeline` runs the full pipeline from an input CSV to a compiled HDF5 file ready for finetuning. It executes three steps in sequence:

1. **Stage 1 (PenCL inference)** -- generates joint text/protein embeddings from the input CSV.
2. **Stage 2 (Facilitator sampling)** -- maps text embeddings into protein embedding space.
3. **HDF5 compilation** -- packages the Facilitator output into HDF5 format.

> **CLI argument reference:** see [CLI_reference.md#biom3_embedding_pipeline--end-to-end-embedding-pipeline](../CLI_reference.md#biom3_embedding_pipeline--end-to-end-embedding-pipeline) for the full argument table.

### Usage

```bash
biom3_embedding_pipeline \
    -i data/dataset_with_prompts.csv \
    -o outputs/embeddings \
    --pencl_weights weights/PenCL/PenCL_V09152023_last.ckpt \
    --facilitator_weights weights/Facilitator/Facilitator_MMD15.ckpt/last.ckpt \
    --pencl_config configs/inference/stage1_PenCL.json \
    --facilitator_config configs/inference/stage2_Facilitator.json \
    --prefix my_dataset
```

### Required arguments

| Flag | Description |
|------|-------------|
| `-i`, `--input_data_path` | Input CSV with sequences and text prompts |
| `-o`, `--output_dir` | Directory for all output files |
| `--pencl_weights` | Path to PenCL model weights (`.ckpt` or `.bin`) |
| `--facilitator_weights` | Path to Facilitator model weights |
| `--pencl_config` | Stage 1 JSON config file |
| `--facilitator_config` | Stage 2 JSON config file |
| `--prefix` | Filename prefix for output files |

### Optional arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cuda` | Device for inference (`cpu`, `cuda`, `xpu`) |
| `--batch_size` | `256` | Batch size for Stage 1 |
| `--num_workers` | `0` | Dataloader workers for Stage 1 |
| `--cross_comparison_sample_limit` | `0` | Stage 1 O(n²) cross-comparison metrics: `0` skips them, `-1` uses all, positive uses that many. Print-only; embeddings unaffected |
| `--no_amp` | off | Run Stage 1 in fp32 instead of autocast (bf16 on xpu) |
| `--float32_matmul_precision` | config (`high`) | Stage 1 fp32 matmul precision; pair `highest` with `--no_amp` for a deterministic fp32 pass |
| `--mmd_sample_limit` | `1000` | Sample limit for MMD in Stage 2 |
| `--dataset_key` | `MMD_data` | HDF5 group name |

### Output files

Given `--output_dir outputs --prefix mydata`, the pipeline produces:

- `outputs/mydata.PenCL_emb.pt` -- Stage 1 embeddings
- `outputs/mydata.Facilitator_emb.pt` -- Stage 2 embeddings
- `outputs/mydata.compiled_emb.hdf5` -- final HDF5 for finetuning
- `outputs/mydata.build_manifest.json` -- arguments, weights, and configs used
- `outputs/mydata.run.log` -- the run's console output

## Standalone HDF5 Compilation

If you have already run Stages 1 and 2 separately, use `biom3_compile_hdf5` to compile the Facilitator output into HDF5:

```bash
biom3_compile_hdf5 \
    -i outputs/mydata.Facilitator_emb.pt \
    -o outputs/mydata.compiled_emb.hdf5 \
    --dataset_key MMD_data
```

| Flag | Description |
|------|-------------|
| `-i`, `--input_data_path` | Facilitator output `.pt` file |
| `-o`, `--output_path` | Output HDF5 file path |
| `--dataset_key` | HDF5 group name (default: `MMD_data`) |
