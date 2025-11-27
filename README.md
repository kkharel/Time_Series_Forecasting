# Time Series Forecasting: Global Land Surface Temperature
**Author:** Kushal Kharel  

## Project Overview

This project implements and critically compares both traditional statistical methods (**SARIMA**) and modern deep learning architectures (**RNN**, **GRU**) for forecasting the Global Average Land Surface Temperature.

The core objective was to perform a definitive, standardized **12-step forecast comparison**, rigorously evaluating two multi-step forecasting strategies:

### 1. Recursive Multi-Step Forecasting
A model predicts one step ahead and its output is recursively fed back into the model for 12 total steps.

### 2. Direct Multi-Output Forecasting (Seq2Seq)
A model predicts all 12 steps simultaneously in a single forward pass.

The time series was pre-processed using **differencing and scaling** to ensure stationarity—critical for stable model training.

---

## Key Findings Summary

### 1. Statistical Dominance

**Manual ARIMA (MAE: 0.2081)** is the overall best model.  
For seasonal, stationary, and well-structured time series, a well-tuned classical method still outperforms deep learning models.  
The benchmark MAE of **0.2081** was unmatched by any neural network architecture.

---

### 2. Validating the Direct Multi-Step Approach

Removing compounding error significantly improved deep learning performance:

- **GRU Encoder–Decoder (MAE: 0.3380)** is the best deep learning model.  
- Predicting all 12 steps at once reduces error accumulation.
- **GRU (Simple Seq2Seq) ** performed the worst, suffering from information bottleneck.

---

### 3. Complexity Finally Pays Off

- GRU Encoder–Decoder (0.3380) outperformed all simpler RNN variants.
- Stacking two GRU layers with `return_sequences=True` preserved temporal patterns more effectively.
- This avoided the bottleneck created by Dense(12) output heads used in simpler Seq2Seq models.

---

## Conclusion

This experiment demonstrates two important time series principles:

### 1. Classical Models for Structured Data
A well-tuned **SARIMA** model remains unbeatable when seasonality and trend are strong.

### 2. Architecture Over Model Choice
For deep learning:

- **Direct Multi-Step (Seq2Seq)** > **Recursive (Seq2Vec)**
- GRU achieves its best results only when implemented within a full **Encoder–Decoder** sequence model.

---

## Model Performance Summary (Mean Absolute Error)

| Model Type     | Architecture               | Prediction Method            | 12-Step MAE | Performance Ranking |
|----------------|----------------------------|------------------------------|-------------|----------------------|
| Classical      | Manual ARIMA               | Direct (Statistical)         | **0.2081**  | **1st (Best)**       |
| Baseline       | Naive Forecast             | Direct                       | 0.2740      | 2nd                  |
| Classical      | Auto ARIMA                 | Direct (Statistical)         | 0.2923      | 3rd                  |
| Deep Learning  | GRU (Encoder-Decoder)      | Direct Multi-Step (Seq2Seq)  | 0.3393      | 4th (Best DL)        |
| Deep Learning  | Simple RNN (Recursive)     | Recursive (Seq2Vec)          | 0.3434      | 5th                  |
| Deep Learning  | Linear NN (Seq2Seq)        | Direct Multi-Step            | 0.3513      | 6th                  |
| Deep Learning  | Linear NN                  | Recursive (Seq2Vec)          | 0.3531      | 7th                  |
| Deep Learning  | Simple RNN (Simple Seq2Seq)| Direct Multi-Step            | 0.3557      | 8th                  |
| Deep Learning  | GRU (Recursive)            | Recursive (Seq2Vec)          | 0.3674      | 9th                  |
| Deep Learning  | GRU (Simple Seq2Seq)       | Direct Multi-Step            | 0.3839      | **10th (worst)**     |

---

# Methodology and Architectural Insights

## 1. Data Preparation

### Differencing
To remove trend and seasonality, the temperature series was:

- Differenced once (non-seasonal: `d = 1`)
- Seasonally differenced once (seasonal: `D = 1`)

ACF/PACF confirmed stationarity after differencing.

### Windowing
A fixed input window of **12 months** was used for all deep learning models to match the seasonal cycle.

---

## 2. Deep Learning Architectural Tests

### A. Recursive Multi-Step Forecasting

**Strategy:**  
Train the model for `Y[t+1]`, feed prediction back for steps `t+2` to `t+12`.

**Result:**  
- **Best recursive model:** Simple RNN 
- Simpler architectures reduce compounding error.

---

### B. Direct Multi-Output GRU (Simple Seq2Seq)

**Architecture:**

RNN/GRU → return_sequences=False → Dense(12)

**Limitation:**  
This created a severe **information bottleneck**, forcing all future steps to be encoded in a single hidden state.

---

### C. Sequence-to-Sequence (Encoder–Decoder)

**Architecture:**

GRU → GRU → TimeDistributed(Dense(1))


**Success:**  
- Eliminated the bottleneck  
- Enabled step-specific prediction  
- Achieved **best deep learning performance (MAE: 0.3393)**

---

# Project Files

| File | Description |
|------|-------------|
| **Time_Series_Forecasting.ipynb** | Main notebook containing data loading, SARIMA analysis, deep learning models, and evaluation. |

---

# Dependencies

This project requires:

- Python **3.8+**
- **TensorFlow / Keras**
- **NumPy**
- **Pandas**
- **Scikit-learn** (scaling, MAE)
- **Statsmodels** (SARIMA)
- **pmdarima** (Auto ARIMA)

