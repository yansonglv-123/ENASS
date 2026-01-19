# ENASS
Energy-Aware Single-Source Progressive Generalization for Cross-Scene mmWave Radar Human Activity Sensing
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Journal: Sensors](https://img.shields.io/badge/Submitted%20to-Sensors-blue)](https://www.mdpi.com/journal/sensors)

> Note:This repository contains the inference code and network architecture for the paper submitted to Sensors (MDPI). The source code is provided to verify the reproducibility of the proposed method.

📝 Abstract



<!-- ![Model Architecture](pipeline.png) -->
Download Weights: Please download the pre-trained weights (eval_model.pt) from the Releases Page and place it in the root directory.
 📂 Project Structure

The repository is organized as follows:

```text
├── model.py            # Complete definition of the CNN-LSTM network architecture
├── test.py             # Inference script to demonstrate model execution
├── eval_model.pt       # Pre-trained model weights (for reproducibility)
└── README.md           # Documentation
