# MAD-GNN: Multi-Area Graph Neural Network for Bandgap Prediction
 📌 Overview:
 
 This project implements MAD-GNN (Multi-Area Graph Neural Network) to predict the bandgap of crystalline materials using Graph Neural Networks.
 
 🧠 Motivation:
 
Traditional GNNs treat all edges similarly. However, in materials science, different interactions 
- Local interactions (B–X)
- Ionic interactions (A–X / A–B)
- Long-range interactions
have distinct physical significance.

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
	
	Project Structure
	data/
    models/
    utils/
    train.py
    evaluate.py
    requirements.txt

    <img width="919" height="740" alt="Unknown-5" src="https://github.com/user-attachments/assets/d64fc0bd-00b1-4faa-8a35-0b92dff1913f" />
📸 Sample Output


	
  📊 Results
  Improved prediction performance compared to baseline GNN models
  
  Evaluated using Mean Absolute Error (MAE)

	
  ▶️ How to Run
  pip install -r requirements.txt
  
  python train.py


  


	
  
