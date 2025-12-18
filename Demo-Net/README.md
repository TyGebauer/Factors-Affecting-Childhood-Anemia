# Dataset Preprocessing Overview

> **Environment Note:**  
> Use the **Demo-Net project root folder** as the working environment for all preprocessing and training commands.

This project uses three datasets for training **DEMO-Net** and baseline **GNN models**:

- Facebook Page–to–Page Graph  
- USA Airport Connectivity Graph  
- ENZYMES Dataset  

Each dataset goes through a preprocessing pipeline that converts raw graphs or files into **PyTorch Geometric–ready tensors**, stored as `processed.pt`. These processed files are what the training scripts load during execution.

---

## Facebook Dataset (`facebook_paper_degX_binsY`)

### Folder Used for Training
- data/facebook_paper_degX_binsY/ processed.pt
#### Raw Files
- facebook_combined.txt (or facebook348.edgelist, depending on the variant)

### Purpose
Transform the Facebook page–to–page graph into:
Degree-based node features (degX)
Binned degree labels or classification targets (binsY)

### Preprocessing Scripts
- prep_facebook_binned_degree.py
- prep_facebook_degfeat_binslabels.py
- prep_facebook_degree_labels.py

### Preprocessing Pipeline
Load the raw edge list and construct an undirected graph
Compute the degree for every node
Create feature matrix X based on node degree or degree-derived statistics
Create label vector Y by binning node degrees into categories
Save the resulting PyTorch Geometric Data object as processed.pt

### Training Commands
- python train_nodeclf.py --model gcn --data_path data/facebook_paper_degX_binsY/processed.pt
- python train_nodeclf.py --model demo_net_weight --data_path data/facebook_paper_degX_binsY/processed.pt
- python train_nodeclf.py --model demo_net_hash --data_path data/facebook_paper_degX_binsY/processed.pt

## USA Airports Dataset (usa_airports_degX_binsY)
### Folder Used for Training
- data/usa_airports_degX_binsY/processed.pt

### Raw Files
- usa_airports.txt (graph structure)
Additional airport metadata (e.g., activity levels)

### Purpose
- Transform the airport connectivity network into:
- Degree-based node features
- Activity-bin node labels (binsY)

### Preprocessing Scripts
- prep_usa_airports.py
- prep_usa_airports_feats.py
- prep_usa_airports_activity_labels.py

### Preprocessing Pipeline
- Load the airport graph (directed or undirected, depending on script)
- Compute degree-based node features and store them in X
- Load airport activity metadata
- Bin activity levels into categorical labels Y
- Save the PyTorch Geometric Data object as processed.pt

### Training Commands
- python USA_train_nodeclf.py --model gcn --data_path data/usa_airports_degX_binsY/processed.pt
- python USA_train_nodeclf.py --model demo_weight --data_path data/usa_airports_degX_binsY/processed.pt
- python USA_train_nodeclf.py --model demo_hash --data_path data/usa_airports_degX_binsY/processed.pt
  
## ENZYMES Dataset
## Folder Used for Training

- data/ENZYMES_raw/ raw graph files for 600 enzyme graphs

## Purpose
-Load the standard 6-class ENZYMES graph classification dataset, commonly used in GNN benchmark studies.

## Preprocessing
- Preprocessing is handled internally within the training script: train_enzymes.py

## Preprocessing Pipeline
- Load raw ENZYMES graphs using torch_geometric.datasets.TUDataset
- Apply node feature normalization when required
- Format each graph with:
    - Node feature matrix
    - Adjacency structure
    - Graph-level class label
- Cache processed graphs for faster future runs

## Training Commands
- python train_enzymes.py --model deepwl
- python train_enzymes.py --model demo_hash
- python train_enzymes.py --model demo_weight
