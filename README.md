# Time Series Forecasting: Global Land Surface Temperature

**Author:** Kushal Kharel  

## Project Overview

This project rigorously evaluates classical statistical and modern deep learning approaches for forecasting Global Land Surface Temperature. The primary goal was a **standardized comparison of 12-step forecast strategies**, across multiple architectures and feature representations.

Two multi-step forecasting strategies were tested:

1. **Recursive Multi-Step Forecasting (Seq2Vec)**  
   Predicts one step ahead recursively, feeding each prediction into the next step.  

2. **Direct Multi-Output Forecasting (Seq2Seq)**  
   Predicts all 12 steps simultaneously in a single forward pass, capturing horizon-wide dependencies.

The time series was pre-processed using **differencing** to remove trend and seasonality, followed by **MinMax scaling**. Seasonal features (sine and cosine of month) were included in multivariate experiments.

---

## Key Insights

### 1. Single-Step Forecasting
- **Manual ARIMA (MAE: 0.2081)** is the most accurate single-step model.  
- Linear regression and naive forecasts remain competitive due to strong autocorrelation and seasonality.  
- Deep learning models trained recursively (RNN/GRU) slightly underperform, as differencing removes the long-term dependencies that GRUs are designed to capture.

### 2. Multi-Step Direct Forecasting
- **GRU models (encoder-decoder and last hidden state)** outperform RNNs and linear models for 12-step forecasts.  
- Surprisingly, multi-step GRUs slightly outperform single-step GRUs because direct multi-step training **reduces cumulative error** and captures joint temporal dependencies.  
- Simple RNNs remain competitive, showing that simpler recurrent architectures effectively model short-to-medium horizon sequences.

### 3. Multivariate Features Provide Incremental Gains
- Including **sine and cosine seasonal features** improves multi-step forecast performance for GRU models.  
- The multivariate encoder-decoder GRU achieved the **best overall deep learning MAE** (0.3353), slightly better than its univariate counterpart.

### 4. Linear Models vs Neural Networks
- Linear models perform adequately for single-step predictions but are less effective for multi-step forecasting.  
- Neural networks, especially GRUs, can model nonlinear interactions across multiple steps, improving accuracy on direct 12-step forecasts.

---

## Model Performance Summary (Mean Absolute Error)

| Model Type         | Architecture                                | Prediction Method            | MAE        | Notes |
|-------------------|---------------------------------------------|-----------------------------|------------|-------|
| Classical          | Manual ARIMA                                | Single-Step                 | **0.2081** | Best overall |
| Baseline           | Naive Forecast                               | Single-Step                 | 0.2740     | Strong simple baseline |
| Classical          | Auto ARIMA                                   | Single-Step                 | 0.2923     | Statistical baseline |
| Deep Learning      | GRU (Encoder–Decoder, univariate)           | Multi-Step Direct (12-step) | 0.3363     | Captures horizon dependencies |
| Deep Learning      | GRU (Encoder–Decoder, multivariate)         | Multi-Step Direct (12-step) | 0.3353     | Best DL with seasonal features |
| Deep Learning      | GRU (Last Hidden State)                     | Multi-Step Direct (12-step) | 0.3359     | Comparable to encoder-decoder |
| Deep Learning      | Simple RNN (Last Hidden State)              | Multi-Step Direct (12-step) | 0.3364     | Close to GRU performance |
| Deep Learning      | Linear Model                                | Multi-Step Direct (12-step) | 0.3415     | Baseline for neural comparison |
| Deep Learning      | Simple RNN (Single-Step Recursive)          | Recursive (1-step)          | 0.3547     | Accumulates error recursively |
| Deep Learning      | GRU (Single-Step Recursive)                 | Recursive (1-step)          | 0.3652     | Higher recursive error |
| Deep Learning      | Linear Model (Single-Step)                  | Single-Step                 | 0.3486     | Linear baseline for single-step |

---

## Methodology and Architectural Insights

### Data Preparation
- **Differencing:** Remove trend (`d=1`) and seasonality (`D=1`)  
- **Scaling:** MinMax scaling after differencing  
- **Windowing:** 12-month input sequences  
- **Seasonal Features:** Sine and cosine transformations for multivariate experiments

### Deep Learning Architectural Strategies

**Recursive Multi-Step (Seq2Vec):**  
- Single-step forecast recursively fed into subsequent steps  
- Simpler architectures (Simple RNN) mitigate error accumulation  

**Direct Multi-Step (Seq2Seq):**  
- Predicts all 12 steps simultaneously  
- GRU models, especially encoder-decoder, eliminate information bottlenecks  
- Multivariate features slightly improve accuracy

---

## Conclusion

- **Single-Step Forecasting:** Classical ARIMA dominates, leveraging seasonality and trend explicitly.  
- **Multi-Step Forecasting:** GRUs outperform linear models, capturing long-horizon dependencies and benefiting from seasonal features.  
- **Model Selection Should Reflect Horizon and Data:** Recursive single-step favors simpler statistical models; direct multi-step favors GRUs with feature-enriched inputs.  
- **Practical Insight:** Multi-step direct training can outperform recursive single-step training by reducing cumulative error and capturing joint temporal dependencies.

---

### Before concluding, there is a fundamental part missing in this study that is hyperparameter tuning. The immediate next step will be hyperparamter tuning for the best performing NN model

## Project Files

| File | Description |
|------|-------------|
| **Time_Series_Forecasting.ipynb** | Notebook containing data loading, ARIMA analysis, deep learning models, and evaluation |

---

## Dependencies

- Python 3.8+  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Scikit-learn  
- Statsmodels  
- pmdarima
