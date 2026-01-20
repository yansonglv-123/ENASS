# ENASS
Energy-Aware Single-Source Progressive Generalization for Cross-Scene mmWave Radar Human Activity Sensing
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Journal: Sensors](https://img.shields.io/badge/Submitted%20to-Sensors-blue)](https://www.mdpi.com/journal/sensors)

> Note:This repository contains the inference code and network architecture for the paper submitted to Sensors (MDPI). The source code is provided to verify the reproducibility of the proposed method.

Download Weights: Please download the pre-trained weights (eval_model.pt) from the Releases Page and place it in the root directory.

📅 Data Availability(https://aiotgroup.github.io/XRF55/)

📂 Project Structure 

The repository is organized as follows:

```text
├── model.py            # Complete definition of the CNN-LSTM network architecture
├── test.py             # Inference script to demonstrate model execution
├── test2.py            # Inference script for EDL
├── eval_model.pt       # Pre-trained model weights (for reproducibility)
└── README.md           # Documentation
```
Result：

<img width="40%" alt="屏幕截图 2026-01-19 213010" src="https://github.com/user-attachments/assets/4d192a62-c94a-428c-b744-11aa154fe049" />
<img width="40%"  alt="pycharm64_KwTEJ59y3q" src="https://github.com/user-attachments/assets/8084ff81-19f8-4901-8d5a-e4f8a5c4d554" />
<img width="40%"  alt="pycharm64_eL7ObIH4PV" src="https://github.com/user-attachments/assets/8547ae4e-c407-4870-9b8c-7b529f2a1309" />
<img width="40%"  alt="pycharm64_5nfUy3WWH9" src="https://github.com/user-attachments/assets/89267b05-c5f3-4b23-935c-3fbc31302d4e" />
<img width="40%"  alt="pycharm64_03u8N7LyL9" src="https://github.com/user-attachments/assets/b8fb9aca-e040-440b-af09-8ef44d065e67" />
