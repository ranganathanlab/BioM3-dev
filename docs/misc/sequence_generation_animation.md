# Sequence Generation Animation

`biom3_ProteoScribe_sample` can produce GIF animations that visualise the
ProteoScribe diffusion denoising process frame-by-frame. Each frame corresponds
to one step of the absorbing-state diffusion, showing how the model progressively
fills in masked positions (`-`) with amino acid characters to build the final
sequence.

## How it works

ProteoScribe uses an absorbing-state discrete diffusion model. During sampling,
the sequence starts fully masked and positions are unmasked one at a time in a
random order determined by a permutation of the diffusion steps. After every
step the current partial sequence is recorded; these snapshots become the frames
of the GIF.

## CLI flags

Three optional flags control animation. No flags are required; omitting
`--animate_prompts` disables the feature entirely.

| Flag | Values | Default | Description |
| ---- | ------ | ------- | ----------- |
| `--animate_prompts` | space-separated integers, `all`, or `none` | *(omit to disable)* | Which prompt indices to animate |
| `--animate_replicas` | integer `i`, `all`, or `none` | `1` | How many replicas to animate per prompt; `i` selects `range(0, i)` |
| `--animation_dir` | path | `<output_dir>/animations/` | Directory where GIFs are written |

### `--animate_prompts`

Controls which of the input prompts (rows of `z_c`) are animated.

```bash
--animate_prompts 0          # only prompt 0
--animate_prompts 0 1 2      # prompts 0, 1, and 2
--animate_prompts all        # every prompt in the embedding file
--animate_prompts none       # no animation (same as omitting the flag)
```

Requesting a specific index that does not exist raises an error with the valid
range.

### `--animate_replicas`

Controls how many replicas are animated for each selected prompt. Replicas are
selected from index 0 up to (not including) the requested count.

```bash
--animate_replicas 1         # replica 0 only (default when animating)
--animate_replicas 3         # replicas 0, 1, 2
--animate_replicas all       # every replica (up to num_replicas)
--animate_replicas none      # no animation
```

If the requested count exceeds `num_replicas` (from `--num_replicas`, else the
JSON config), it is clamped to `num_replicas` and a warning is logged.

### `--animation_dir`

```bash
--animation_dir outputs/my_animations
```

Defaults to `animations/` inside the same directory as `--output_path`. The
directory is created if it does not exist.

## Output format

One GIF is produced per (prompt, replica) pair:

```
<animation_dir>/
    prompt_0_replica_0.gif
    prompt_0_replica_1.gif
    prompt_1_replica_0.gif
    ...
```

Each GIF has `diffusion_steps` frames (set in the JSON config; typically 1024).
Unsampled positions are shown as `-`; `<START>` and `<END>` tokens appear
literally at the sequence boundaries.

## Examples

### Animate one prompt, one replica (minimal)

```bash
biom3_ProteoScribe_sample \
    --input_path  outputs/facilitator_embeddings.pt \
    --config_path   configs/inference/stage3_ProteoScribe_sample.json \
    --model_path  weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --output_path outputs/generated_sequences.pt \
    --animate_prompts 0
```

Produces `outputs/animations/prompt_0_replica_0.gif`.

### Animate a subset of prompts, multiple replicas

```bash
biom3_ProteoScribe_sample \
    --input_path  outputs/facilitator_embeddings.pt \
    --config_path   configs/inference/stage3_ProteoScribe_sample.json \
    --model_path  weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --output_path outputs/generated_sequences.pt \
    --animate_prompts 0 1 2 \
    --animate_replicas 3
```

Produces nine GIFs: `prompt_{0,1,2}_replica_{0,1,2}.gif`.

### Animate everything, custom output directory

```bash
biom3_ProteoScribe_sample \
    --input_path    outputs/facilitator_embeddings.pt \
    --config_path     configs/inference/stage3_ProteoScribe_sample.json \
    --model_path    weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --output_path   outputs/generated_sequences.pt \
    --animate_prompts all \
    --animate_replicas all \
    --animation_dir outputs/full_animation
```

## Performance note

Collecting animation frames adds negligible overhead — the intermediate
per-step sequence arrays are already produced by the sampling loop and are
captured in CPU memory only for the selected (prompt, replica) pairs. GIF
encoding runs after sampling completes. For large runs, prefer targeting
specific prompts rather than `all` to keep memory use bounded.

## See also

- `demos/animate_generation.sh` — runnable end-to-end demo
- `src/biom3/Stage3/animation_tools.py` — GIF rendering utilities
- `docs/gumbel_max_sampling.md` — details of the sampling step used inside the loop
