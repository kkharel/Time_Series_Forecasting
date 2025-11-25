# Time Series Forecasting: Global Land Surface Temperature

## Project Overview

This project implements and critically compares both traditional statistical methods ($\text{SARIMA}$) and modern deep learning architectures ($\text{RNN}$, $\text{GRU}$) for forecasting the Global Average Land Surface Temperature.The core objective was to perform a definitive, standardized 12-step forecast comparison, testing two major deep learning strategies:

Recursive Multi-Step Forecasting: Training a model for one step and recursively rolling its prediction forward for 12 steps.

Direct Multi-Output Forecasting (Seq2Seq): Training a model to predict all 12 steps simultaneously.The time series was first pre-processed using differencing and scaling to ensure stationarity, which is critical for model stability

## Key Findings Summary

The project demonstrated that architectural choice and problem framing are more critical than model complexity.

| Rank | Task | Model Configuration | Validation MAE | Conclusion | 
| ----- | ----- | ----- | ----- | ----- | 
| **1** | **1-Step** | **Manual SARIMA** | **0.2647** | **Overall Winner.** Mathematically superior at capturing the underlying linear, seasonal structure. | 
| 2 | **12-Step** | **GRU Encoder-Decoder (Seq2Seq)** | **0.3405** | **Best Deep Learning Multi-Step Model.** The Seq2Seq structure successfully solved the information bottleneck. | 
| 3 | 1-Step | SimpleRNN (Seq2Vec) | 0.2914 | Best performance on the 1-step, stationary data. More efficient than the GRU for short-term correlation. | 
| 4 | 12-Step | SimpleRNN (Seq2Vec) | 0.3472 | Managed to beat the Linear Baseline, but suffered from the information bottleneck. | 
| 5 | 12-Step | Linear Model (Baseline) | 0.3539 | Crucial baseline for the multi-step challenge. | 
| 6 | 12-Step | GRU (Seq2Vec) | 0.4173 | **Worst Performer.** Failed significantly due to the extreme information bottleneck and complex gate interference when predicting 12 steps from a single hidden state. | 

## Methodology and Architectural Insights

### 1. Data Preparation

* **Differencing:** The land temperature series, which exhibited strong trend and seasonality, was **doubly differenced** (seasonal and non-seasonal) to achieve stationarity.

* **Scaling:** Data was scaled using `MinMaxScaler` before feeding into neural networks.

* **Windowing:** A fixed window of **12 months** (`WINDOW_SIZE=12`) was used as input for all models, aligned with the annual seasonal cycle.

### 2. Deep Learning Architectural Test

The project rigorously tested three primary deep learning architectures:

#### A. Sequence-to-Vector (Seq2Vec) for 1-Step

* **Goal:** Predict one single value ($\hat{Y}_{t+1}$) from a 12-step input sequence.

* **Result:** SimpleRNN was the winner, demonstrating that for stationary data, simpler models are often more parsimonious.

#### B. Sequence-to-Vector (Seq2Vec) for 12-Steps (Direct Multi-Step)

* **Architecture:** RNN/GRU $\rightarrow$ `return_sequences=False` $\rightarrow$ `Dense(12)`.

* **Limitation:** This created an **information bottleneck**, forcing the recurrent layer to compress all 12 necessary output signals into a single, fixed-size hidden vector. This caused the GRU's performance to drop drastically. 

#### C. Sequence-to-Sequence (Seq2Seq) Encoder-Decoder for 12-Steps

* **Architecture:** GRU $\rightarrow$ `return_sequences=True` $\rightarrow$ `TimeDistributed(Dense(1))`.

* **Success:** This structure eliminated the bottleneck by allowing the GRU to output a hidden state for every step, mapping each one to a corresponding output prediction. This validated the GRU's effectiveness in a proper sequential forecasting setup. 


## Project Files

| File | Description | 
| ----- | ----- | 
| `Time_Series_Forecasting.ipynb` | Main Jupyter Notebook containing all data loading, SARIMA analysis, deep learning model training, and evaluation. | 

## Dependencies

This project requires the following libraries:

* Python 3.8+

* TensorFlow / Keras

* NumPy

* Pandas

* Scikit-learn (for scaling and MAE calculation)

* Statsmodels (for SARIMA analysis)
