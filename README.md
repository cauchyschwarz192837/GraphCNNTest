# Graph-Convolutional Neural Network for Piezoelectric Coefficient Prediction in Thin Films

This is an implementation of a graph convolutional neural network (CNN) to predict the piezoelectric coefficient, $d_{33}$, of various crystalline materials. Baseline regression models such as Support Vector Regression (SVR), Random Forest and fully connected neural networks (MLP) were implemented for comparative evaluation against the graph-CNN. CIF files contain list of atoms, their coordinates and who is bonded / neighboring whom for various existing materials. Each such material is mapped to a graph by such that each atom gets a feature vector of shape (N_atoms × atom_feature_dim). For each atom pair within a certain cutoff distance, the edge between them has another feature vector. Per material, we also attach process parameters and fabrication conditions (since each thin film is synthesized under different parameters according to reported conditions in the research papers we scanned) as a global feature vector for the whole CIF file.

I then defined a custom collate function for batch construction, since each crystal has a different number of atoms. This collation method can be modified to explore possible improvements in prediction accuracy, as at the time of writing this code, we have not explored other collation methods due to time constraints. Please see "mynewCollate" for details.

$$
\mathbf{h}_i^{(k+1)} =
\mathrm{MLP}\left(
\mathbf{h}_i^{(k)} +
\sum_{j \in \mathcal{N}(i)}
\phi\!\left(\mathbf{h}_i^{(k)}, \mathbf{h}_j^{(k)}, \mathbf{e}_{ij}\right)
\right)
$$


data/id_prop.csv: This file maps CIF filename to target value, $d_{33}$

Example in command line:
  python train.py data/ --epochs 60 --batch-size 20 --lr 0.01 --val-ratio 0.2 --test-ratio 0.2

CIFData scans data/cif/*.cif, reads id_prop.csv, matches the CIF file to target $d_{33}, then builds an atom graph from the CIF file, returning atom features, neighbor features, adjacency indices, sputter features, crystal index map and target value.


