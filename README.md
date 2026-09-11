# Gonzalez-Buitron-et-al_RNA-Ligands_2026
Code and data supporting the paper "Exploring RNA conformational diversity and RNA-ligand interaction using CoDNaS-RNA".

## **Requirements**
- UV (>=0.8)
- RING 4.0 (see [RING download](https://ring.biocomputingup.it/download))

### **How to install** `uv`?
Go to the [UV installation guide](https://docs.astral.sh/uv/getting-started/installation/)

## **Installation**
### Step 1
```bash
git clone <repository-url>; cd <repository-name>
```

### Step 2 (this will create a virtual environment and install the dependencies)
```bash
uv sync
```

## **Usage**
After cloning the repository and installing the dependencies, go to the repository folder
```bash
cd <repository-name>
```

and activate the virtual environment:
```bash
source .venv/bin/activate
```

### **Working with the notebooks**

#### **Steps to reproduce figures and tables**
You will need the tables present in `./dataset/processed/`. These are:

- `initial_cluster_stats.tsv.gz`
- `initial_conformer_stats.tsv.gz`
- `initial_dataset_clusters.tsv.gz`
- `initial_dataset_conformers.tsv.gz`
- `initial_dataset_pairs.tsv.gz`
- `initial_dataset_residues.tsv.gz`
- `initial_dataset_residues_w_interactions.tsv.gz`
- `ligands_w_threshold.tsv.gz`

- `final_dataset_clusters.tsv.gz`
- `final_dataset_conformers.tsv.gz`
- `final_dataset_pairs.tsv.gz`
- `final_ligands_w_threshold.tsv.gz`
- `final_conformer_stats.tsv.gz`
- `final_cluster_stats.tsv.gz`

- `rmsd_c3_prime_per_position.tsv.gz`

Go to the `src/notebooks` folder and run the following notebooks in order:
- figures_and_tables.ipynb

**Using Jupyter Lab or from VS Code.**
```bash
jupyter lab src/notebooks/figures_and_tables.ipynb
```
or
```bash
code src/notebooks/figures_and_tables.ipynb
```

## **Data sources**

### CoDNaS-RNA
- Website: [CoDNaS-RNA](http://ufq.unq.edu.ar/codnasrna/)
- Release: 2026-01
- Description: All released tables from CoDNaS-RNA.
- Dirpath: [codnas-rna](./databases/codnas-rna/releases/2026-01/)