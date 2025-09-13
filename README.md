# Pseq2Sites

**Pseq2Sites** is a sequence-based deep learning model designed for **protein ligand binding site prediction**. It predicts the specific amino acid residues where a small-molecule ligand is likely to bind, using only the protein's amino acid sequence as input. This approach bypasses the need for complex 3D structural data, making it a valuable tool in bioinformatics and drug discovery.

The model was trained on the **PDBbind** and **sc-PDB** datasets, and its predictive performance was rigorously evaluated using independent benchmark datasets including **COACH420**, **HOLO4K**, and **CSAR**.

-----

## Installation

To set up the environment, we recommend using Conda or Miniconda. Simply run the following command to create a new environment with all the necessary dependencies:

```bash
conda env create -f environment.yml
```

-----

## Getting Started: A Step-by-Step Guide

The training and evaluation process is organized into a series of Jupyter notebooks, making it easy to follow along. The notebooks are sequentially numbered to guide you through each step.

### Step 1. Get Protein Sequences and Binding Sites from 3D Structural Data

This step involves preparing the data from publicly available databases. The notebook `1_Preprocess_Data_PDBbind2020R1.ipynb` demonstrates how to parse protein sequences and binding site information from the **PDBbind v2020R1** dataset.

**Required Data:**

  * PDBbind: [https://www.pdbbind-plus.org.cn/](https://www.pdbbind-plus.org.cn/)
  * scPDB: [http://bioinfo-pharma.u-stbg.fr/scPDB/](https://www.google.com/search?q=http://bioinfo-pharma.u-stbg.fr/scPDB/)
  * COACH420, HOLO4K: [https://github.com/rdk/p2rank-datasets](https://github.com/rdk/p2rank-datasets)

The output, including lists of proteins and preprocessed PDB files, will be saved in the `data_preprocessed` directory.

### Step 2. Generate Protein Feature Vectors

Proteins are represented as numerical vectors for the deep learning model. The `2_Protein_Feature_Generation.ipynb` notebook uses the pre-trained **ProtTrans** (specifically, the T5-Encoder model) to generate 1024-dimensional embedding vectors from the protein sequences. These embeddings will be stored in the `data_embeddings` directory.

### Step 3. Train the Pseq2Sites Model

With the preprocessed data and embeddings ready, you can now train the model. Run `3_Train.ipynb` to begin training. The model is trained using **5-fold cross-validation**, and the trained model checkpoints (`.pth` files) will be saved in the `outputs` directory.

### Step 4. Analyze Training Loss

To monitor the training process, the `4_Loss_Analysis.ipynb` notebook allows you to visualize the loss values for each fold. This helps in understanding the model's convergence and stability.

### Step 5. Evaluate Predictive Performance

The final step is to assess the model's performance on the test datasets. The `5_Evaluate_Test.ipynb` notebook summarizes the predictive performance of the Pseq2Sites model trained with 5-fold cross-validation.

-----

## Citation

If you find this work useful for your research, please cite the following paper:

```
@article{seo2024pseq2sites,
  title={Pseq2Sites: enhancing protein sequence-based ligand binding-site prediction accuracy via the deep convolutional network and attention mechanism},
  author={Seo, Sangmin and Choi, Jonghwan and Choi, Seungyeon and Lee, Jieun and Park, Chihyun and Park, Sanghyun},
  journal={Engineering Applications of Artificial Intelligence},
  volume={127},
  pages={107257},
  year={2024},
  publisher={Elsevier}
}
```

-----

## Contacts

For any questions or feedback, please contact the author:

**Prof. Jonghwan Choi**
`jonghwanc@hallym.ac.kr`