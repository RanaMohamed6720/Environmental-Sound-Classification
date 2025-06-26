# Environmental Sound Classification

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Librosa-0.10.0%2B-A60000?logo=librosa&logoColor=white" alt="Librosa">
  <img src="https://img.shields.io/badge/Dataset-UrbanSound8K-green" alt="UrbanSound8K">
  <img src="https://img.shields.io/badge/Models-RNN%7CCNN--LSTM%7CTransformer-magenta" alt="Models">
</div>

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Audio Preprocessing Pipeline](#3-audio-preprocessing-pipeline)
4. [Feature Extraction with Librosa](#4-feature-extraction-with-librosa)
5. [Training Workflow](#5-training-workflow)
6. [Models' Architectures Compared](#6-models-architectures-compared)
7. [Performance Showdown](#7-performance-showdown)
8. [Dependencies](#8-dependencies)
9. [Authors](#9-authors)

## 1. Project Overview
The goal of this project is to implement and compare deep learning models for classifying environmental sounds using the UrbanSound8K dataset. We evaluate RNNs, LSTMs, Transformers, and a novel CNN-LSTM hybrid architecture.

```mermaid
graph TD
    A[UrbanSound8K] --> B(Preprocessing)
    B --> C[Feature Extraction]
    C --> D{Model Comparison}
    D --> E[RNN/LSTM]
    D --> F[Transformer]
    D --> G[CNN-LSTM]
    G --> H[Best Model]
```

## 2. Dataset 
**UrbanSound8K Dataset**: 8732 real-world audio clips across 10 environmental categories

```mermaid
pie
    title Sound Class Distribution
    "air_conditioner" : 1000
    "car_horn" : 429
    "children_playing" : 1000
    "dog_bark" : 1000
    "drilling" : 1000
    "engine_idling" : 1000
    "gun_shot" : 374
    "jackhammer" : 1000
    "siren" : 929
    "street_music" : 1000
```

## 3. Audio Preprocessing Pipeline
```mermaid
graph LR
A[Raw Audio] --> B[Resampling to 16kHz]
B --> C[Mono Conversion]
C --> D[4s Duration Standardization]
D --> E[Silence Padding/Trimming]
E --> F[Feature Extraction]
```

1. **Resampling** 
   All audio → 16,000 Hz (Nyquist theorem compliant: captures up to 8,000 Hz)
   
2. **Mono Conversion**  
   Multi-channel → Single channel for efficient processing
   
3. **Duration Standardization**  
   Fixed 4-second clips:
   - Short clips: Padded with silence
   - Long clips: Intelligently trimmed

## 4. Feature Extraction with Librosa
**Three audio representations** extracted using Librosa:

| Feature | Purpose | Visualization |
|---------|---------------|--------------|
| **Mel Spectrogram** | Captures frequency patterns similar to human hearing | <img src="https://github.com/user-attachments/assets/155daff6-d7a3-439f-a059-40abf0dbbf26" width="150"> |
| **MFCCs** | Compact representation of spectral characteristics | <img src="https://github.com/user-attachments/assets/e232eb06-5c03-401b-a6bb-16cb66f3c3a0" width="150"> |
| **Energy (RMS)** | Measures loudness dynamics over time | <img src="https://github.com/user-attachments/assets/9380df3d-6ac9-483e-ab56-f440387d4ac2" width="150"> |

**Some Observed Class Characteristics**:
```mermaid
graph LR
    A[Car Horn] -->|Sharp transient start| B(High energy burst)
    C[Drilling] -->|Continuous pattern| D(High frequency)
    E[Children Playing] -->|Dynamic variations| F(Speech/laughter patterns)
```

## 5. Training Workflow
```mermaid
flowchart TB
    A[Raw Audio] --> B[Preprocessing]
    B --> C[Feature Extraction]
    C --> D[Model Initialization]
    D --> E[Training Loop]
    E --> F[Validation]
    F --> G{Performance OK?}
    G -->|Yes| H[Save Model]
    G -->|No| I[Adjust Hyperparameters]
    I --> D
    H --> J[Final Evaluation]
```

## 6. Models' Architectures Compared

### RNN/LSTM Models
```mermaid
graph LR
    A[Input Features] --> B[RNN/LSTM Layer]
    B --> C[Bidirectional Processing]
    C --> D[Element-wise Max Pooling]
    D --> E[Dropout]
    E --> F[Linear Layer]
    F --> G[Output Classes]
```

**RNN Architecture**:
```mermaid
graph TD
    A[Input Sequence] --> B[Embedding Layer]
    B --> C[RNN Cell]
    C --> D["Hidden State (h<sub>t</sub>)"]
    D --> C
    D --> E[Next Step]
    E --> F[Max Pooling]
    F --> G[Dropout 60%]
    G --> H[Linear Layer]
    H --> I[Output]
    
    subgraph RNN Core
        C
        D
    end
    
    style C fill:#900C3F,stroke:#333,stroke-width:2px
    style D fill:green,stroke:#333,stroke-width:2px
```
**LSTM Architecture**:
``` mermaid
graph TD
    A[Input Features] --> B[Input Gate]
    A --> C[Forget Gate]
    A --> D[Output Gate]
    B --> E[Cell State]
    C --> E
    D --> F[Hidden State]
    E --> F
    F --> G[Element-wise Max]
    G --> H[Dropout 60%]
    H --> I[Linear Layer]
    I --> J[Output Classes]
    
    subgraph LSTM Cell
        B
        C
        D
        E
        F
    end
    
    style B fill:#f9860c,stroke:#333
    style C fill:#900C3F,stroke:#333
    style D fill:magenta,stroke:#333
    style E fill:green,stroke:#333
    style F fill:purple,stroke:#333
```

**Hyperparameter Optimization**:
```mermaid
mindmap
  root((Hyperparameters))
    Learning Rate
      0.001
      0.0005
      0.0001667
    Hidden Size
      32
      64
      128
      256
    Layers
      1
      2
      3
      4
    Features
      MEL
      MFCC
      Energy
```

**RNN vs LSTM Comparison**:
```mermaid
graph LR
    A[Input] --> B(RNN)
    A --> C(LSTM)
    
    B --> D["Simple recurrent unit"]
    C --> E["Gates (input, forget, output)"]
    C --> F["Cell state memory"]
    
    D --> G[Vanishing Gradient Problem]
    E --> H[Long-term dependencies]
    F --> H
    
    G --> I[Lower Accuracy]
    H --> J[Higher Accuracy]
    
    style B stroke:#f00,stroke-width:2px
    style C stroke:#0f0,stroke-width:2px
```

### Transformer Model
```mermaid
graph TD
    A[Input Audio] --> B[Feature Extraction]
    B --> C[Linear Projection]
    C --> D[Positional Encoding]
    D --> E[Encoder Block 1]
    E --> F[Encoder Block 2]
    F --> G[...]
    G --> H[Encoder Block N]
    H --> I[Average Pooling]
    I --> J[Linear Classifier]
    J --> K[Output Classes]
    
    subgraph Encoder Block
        E --> E1[Multi-Head Attention]
        E1 --> E2[Add & Norm]
        E2 --> E3[Feed Forward Network]
        E3 --> E4[Add & Norm]
    end
    
    style E1 fill:#f9860c,stroke:#333
    style E2 fill:green,stroke:#333
    style E3 fill:#2896c5,stroke:#333
    style E4 fill:purple,stroke:#333
```

**Hyperparameter Configuration**:
```mermaid
mindmap
  root((Transformer Config))
    model_dim
      128
      256
    num_layers
      3
      6
    num_heads
      4
      8
    d_ff
      1024
      2048
    dropout
      0.1
      0.2
    learning_rate
      0.001
      0.0005
```

### CNN-LSTM Hybrid (Best Model)
```mermaid
graph LR
    A[Input Audio] --> B[CNN Feature Extraction]
    B --> C[Batch Normalization]
    C --> D[ReLU Activation]
    D --> E[Max Pooling]
    E --> F[Reshape for LSTM]
    F --> G[Bidirectional LSTM]
    G --> H[Fully Connected Layers]
    H --> I[Softmax Classification]
```

**Hyperparameter Search**:
```mermaid
mindmap
  root((Hyperparameters))
    Learning Rate
      0.001
      0.0005
      0.0001
    LSTM Size
      128
      256
      512
    LSTM Layers
      1
      2
      3
```

## 7. Performance Showdown
```mermaid
graph LR
    A[Input] --> B(CNN)
    A --> C(RNN)
    A --> D(Transformer)
    A --> E(CNN-LSTM)
    
    B --> F[Spatial Features]
    C --> G[Temporal Features]
    D --> H[Global Context]
    E --> I[Spatial + Temporal]
    
    I --> J[Best Performance]
    
    style E stroke:#f00,stroke-width:3px
    style J fill:green,stroke:#333
```

| Model         | Feature   | Test Accuracy | F1-Score | Confused Class     |
|---------------|-----------|---------------|----------|--------------------|
| RNN           | Energy    | 49.34%        | 0.5071   | Jackhammer         |
| RNN           | **MEL**   | **66.00%**    | **0.6964** | Air Conditioner    |
| RNN           | MFCC      | 59.41%        | 0.6002   | Air Conditioner    |
| Transformer   | Energy    | 35.63%        | 0.3161   | Drilling           |
| Transformer   | **MEL**   | **66.42%**    | **0.6919** | Jackhammer         |
| Transformer   | MFCC      | 64.61%        | 0.6520   | Air Conditioner    |
| CNN-LSTM      | Energy    | 51.60%        | 0.52     | Air Conditioner    |
| CNN-LSTM      | **MEL**   | **🌟 76.95% 🌟** | **0.77** | Air Conditioner    |
| CNN-LSTM      | MFCC      | 69.39%        | 0.71     | Drilling           |

**Confusion Matrix (CNN-LSTM with MEL)**:
![image](https://github.com/user-attachments/assets/da795e45-4716-4903-8b96-0c303e8a7fa7)

**Key Findings**:
```mermaid
graph LR
    A[Mel Spectrograms] -->|Best Features| B(76.95% Accuracy)
    C[CNN-LSTM Hybrid] -->|Best Architecture| B
    D[Air Conditioner] -->|Most Challenging| E(Classification)
```

## 8. Dependencies
```mermaid
graph LR
    A[Core] --> B[Python 3.8+]
    A --> C[PyTorch 1.12+]
    A --> D[Librosa]
    E[Visualization] --> F[Matplotlib]
    E --> G[Seaborn]
    E --> H[Plotly]
    I[Utilities] --> J[NumPy]
    I --> K[Pandas]
    I --> L[Soundata]
```

```bash
pip install torch==1.12.0 librosa==0.10.0 matplotlib seaborn plotly numpy pandas soundata
```

## 9. Authors
- Hager Ashraf Mohamed Melook  
- Nouran Ashraf Yousef  
- Rana Mohamed Ali Attia
