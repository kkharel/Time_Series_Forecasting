# Time Series Forecasting: Global Land Surface Temperature

## Project Overview

This project implements and critically compares both traditional statistical methods ($\text{SARIMA}$) and modern deep learning architectures ($\text{RNN}$, $\text{GRU}$) for forecasting the **Global Average Land Surface Temperature**.

The core objective was to perform a definitive, standardized **12-step forecast** comparison, rigorously evaluating two major multi-step strategies:
1.  **Recursive Multi-Step Forecasting:** Training a model for one step and recursively feeding its prediction back for 12 steps.
2.  **Direct Multi-Output Forecasting (Seq2Seq):** Training a model to predict all 12 steps simultaneously.

The time series was first pre-processed using **differencing and scaling** to ensure stationarity, which is critical for model stability.

***

## Key Findings Summary

The final evaluation, standardized to a **12-step forecast horizon** for all models, revealed that the **SARIMA model** was the most accurate overall, while the **Recursive Simple RNN** proved to be the most efficient deep learning strategy.

| Rank | Model Configuration | Forecasting Strategy | Validation MAE (12-Step) | Key Insight |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Manual SARIMA(1,1,1)x(1,1,1)₁₂** | Direct Forecast | **0.2647** | **Overall Winner.** Mathematically superior at capturing the underlying linear, seasonal structure. |
| **2** | **Simple RNN (32 units)** | **Recursive Multi-Step** | **0.2914** | **Best Deep Learning Strategy.** Recursive prediction outperformed all direct multi-output architectures. |
| 3 | Linear Model | Recursive Multi-Step | 0.3086 | Strong non-recurrent baseline, effective on stationary data. |
| 4 | GRU (32 units) | Recursive Multi-Step | 0.3374 | **Architectural Anomaly.** Underperformed the simpler Simple RNN, suggesting unnecessary complexity for this highly stationary series. |
| 5 | GRU Encoder-Decoder | Direct Multi-Output (Seq2Seq) | 0.3405 | Best direct multi-output model, validating the Seq2Seq structure to solve the bottleneck. |
| 6 | Simple RNN | Direct Multi-Output (Seq2Vec) | 0.3472 | Suffered performance loss due to the information bottleneck. |
| 7 | Linear Model (Baseline) | Direct Multi-Output | 0.3539 | Crucial baseline for the direct multi-step challenge. |

***

## Methodology and Architectural Insights

### 1. Data Preparation

* **Differencing:** The land temperature series, which exhibited strong trend and seasonality, was **doubly differenced** (non-seasonal $d=1$ and seasonal $D=1$) based on the ACF/PACF plots to achieve stationarity.
* **Windowing:** A fixed window of **12 months** (`WINDOW_SIZE=12`) was used as input for all models, aligned with the annual seasonal cycle.

### 2. Deep Learning Architectural Test

The experiment rigorously tested the following core strategies:

#### A. Recursive Multi-Step Forecasting (Simple RNN/GRU/Linear)
* **Strategy:** Model is trained for $Y_{t+1}$, and the output is fed back as input for the next step, simulating a roll-forward over 12 months.
* **Result:** The **Simple RNN was the most effective** in this setting ($\text{MAE}=0.2914$), proving that for the highly stationary, differenced data, the recursive roll-forward strategy with a simpler recurrent unit was the most efficient way to maintain accuracy across the horizon.

#### B. Direct Multi-Output (Seq2Vec)
* **Architecture:** RNN/GRU $\rightarrow$ `return_sequences=False` $\rightarrow$ `Dense(12)`.
* **Limitation:** This created a severe **information bottleneck** , where the recurrent layer had to compress all information needed for 12 unique outputs into a single hidden state, leading to degraded performance for both Simple RNN and GRU.

#### C. Sequence-to-Sequence (Seq2Seq) Encoder-Decoder
* **Architecture:** GRU $\rightarrow$ `return_sequences=True` $\rightarrow$ `TimeDistributed(Dense(1))`.
* **Success:** This structure eliminated the bottleneck by using an output sequence, allowing the GRU to output a hidden state for every step. This resulted in the **best performance among all direct multi-output models ($\text{MAE}=0.3405$)**.

***

## Project Files

| File | Description |
| :--- | :--- |
| `Time_Series_Forecasting.ipynb` | Main Jupyter Notebook containing all data loading, SARIMA analysis, deep learning model training, and evaluation. |

## Dependencies

This project requires the following libraries:

* Python 3.8+
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn (for scaling and MAE calculation)
* Statsmodels (for SARIMA analysis)
* `pmdarima` (for Auto ARIMA)
