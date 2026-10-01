# GFP demo dataset

`gfp_sample_dataset.csv` holds 219 protein sequence and text caption pairs from the
green fluorescent protein family (Pfam PF01353). It covers 73 UniProtKB entries, each
appearing three times. It is the corpus used to finetune the BioM3 ProteoScribe decoder
for GFP, published as a small demo dataset.

Columns: `primary_Accession` (UniProtKB entry name), `protein_sequence`,
`[final]text_caption`.

## Sources and licenses

- Protein sequences and the annotation text in the captions come from UniProtKB
  (<https://www.uniprot.org>), © The UniProt Consortium, licensed under the Creative
  Commons Attribution 4.0 International License
  (<https://creativecommons.org/licenses/by/4.0/>). This dataset is a modified
  selection: entries were chosen by Pfam family, and annotation fields and taxonomic
  lineage were combined into the captions.
- Family membership and family names come from Pfam, now part of InterPro
  (<https://www.ebi.ac.uk/interpro/>), released under CC0.

The UniProt and Pfam releases the dataset was built from are not recorded.
