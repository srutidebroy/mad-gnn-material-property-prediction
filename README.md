# MAD-GNN: Multi-Area Graph Neural Network for Bandgap Prediction
 📌 Overview:
 This project implements MAD-GNN (Multi-Area Graph Neural Network) to predict the bandgap of crystalline materials using Graph Neural Networks.
 
 🧠 Motivation:
Traditional GNNs treat all edges similarly. However, in materials science, different interactions (local, ionic, long-range) have distinct physical significance.

MAD-GNN addresses this by modeling these interactions separately

⚙️ Features
	Multi-branch GNN architecture
	Captures:
	Local interactions (B–X)
	Ionic interactions (A–X / A–B)
	Long-range interactions
	Built using PyTorch Geometric
	Custom graph construction from crystal structures
  🛠️ Tech Stack
  Python
	PyTorch
	PyTorch Geometric
	NumPy, Pandas
  📊 Results
  Improved prediction performance compared to baseline GNN models
	Evaluated using Mean Absolute Error (MAE)
  ▶️ How to Run
  pip install -r requirements.txt
  python train.py
  🚀 Future Work
  •	Hyperparameter tuning
	•	Larger dataset integration
	•	Model deployment (API / web app)
  Step 4: requirements.txt
  torch
  torch-geometric
  numpy
  pandas
  scikit-learn
  matplotlib
