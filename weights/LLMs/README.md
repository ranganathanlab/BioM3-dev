
---
### **`weights/LLMs/README.md`**

# LLMs Pre-trained Weights for Compiling PenCL

This folder contains the pre-trained weights for the **ESM2** and **PubMedBERT** models to compile **PenCL** model (Stage 1 of BioM3). The PenCL model aligns protein sequences and text descriptions to compute joint latent embeddings.

## Downloading Pre-trained Weights

### ESM2 Model
To download the ESM2 (650M parameter) model weights:
```bash
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
wget https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt
```

ESM-2 is MIT licensed; `ESM2_LICENSE.md` in this directory is the notice that has
to travel with any copy we redistribute. PubMedBERT carries its own `LICENSE.md`
inside its model directory for the same reason.

### PubMedBERT Model

Make sure large-file storage capabilities are installed in your environment before cloning HuggingFace model card.
```bash
git lfs install 
```

To download the PubMedBERT model weights:
```bash
git clone https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
```

## Usage
Once available, the pre-trained weights can be loaded as follows:

[Your usage instructions here]

## File Structure
After downloading, your weights directory should contain:
```
weights/
└── LLMs/
    ├── esm2_t33_650M_UR50D.pt
    └── BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/
```

Note: The PubMedBERT download will create a directory containing the model weights and configuration files, while ESM2 downloads as a single file.


