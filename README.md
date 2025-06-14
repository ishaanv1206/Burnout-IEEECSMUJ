# Burnout-IEEECSMUJ

# 🏍️ MotoGP Lap Time Prediction using AutoML

This project leverages **FLAML's AutoML framework** to build a high-performance regression model for predicting **MotoGP lap times** using rich race data. The core ML engine is a tuned **LightGBM Regressor**, automatically optimized through FLAML.

---

## 📊 Problem Statement

Given historical MotoGP data — including rider info, bike and team details, environmental conditions, tire specifications, and race statistics — the objective is to accurately **predict average lap time in seconds** (`Lap_Time_Seconds`).

---

## Main final Code File is final_submission.ipynb/py

## 🧠 Model Summary

- 📦 **Model Type**: Regression  
- ⚙️ **Best Estimator**: `LGBMRegressor`  
- 🧪 **Evaluation Metric**: RMSE (Root Mean Squared Error)  
- 📉 **Best Validation Loss**: ~0.0000  
- 🔍 **AutoML Framework**: FLAML  
- ✅ **Best Learner**: `lgbm`  

**Best Estimator:**  
LGBMRegressor(n_estimators=272, num_leaves=14222, learning_rate=0.10)

## 📁 Dataset Overview

The data is split into three parts:

| File       | Description                             |
|------------|-----------------------------------------|
| `train.csv`| Training set to train the model         |
| `val.csv`  | Validation set for evaluation           |
| `test.csv` | Test set used to generate predictions   |

---

## 🧾 Feature Description

<details>
<summary><strong>Click to expand</strong></summary>

| Column                          | Description |
|---------------------------------|-------------|
| `Unique ID`                     | A unique identifier for each row |
| `Rider_ID`                      | Unique identifier for each rider |
| `category_x`                    | Racing category (MotoGP, Moto2, etc.) |
| `Circuit_Length_km`            | Length of the circuit in kilometers |
| `Laps`                          | Total number of laps in the session |
| `Grid_Position`                | Rider's starting grid position |
| `Avg_Speed_kmh`                | Average speed during the session |
| `Track_Condition`              | Surface state (e.g., dry, wet) |
| `Humidity_%`                   | Humidity during session |
| `Tire_Compound_Front`          | Front tire type |
| `Tire_Compound_Rear`           | Rear tire type |
| `Penalty`                      | Penalty applied (e.g. +3s, DNF) |
| `Championship_Points`          | Total championship points |
| `Championship_Position`        | Current championship standing |
| `Session`                      | Session type (Race, Quali) |
| `Year_x`                       | Year of session |
| `Sequence`                     | Sequence in race |
| `Rider`, `Rider_name`          | Rider info |
| `Team`, `Team_name`            | Team info |
| `Bike`, `Bike_name`            | Bike info |
| `Position`                     | Final race position (negative = DNF/DNS/DSQ/DNQ) |
| `Points`                       | Points awarded for this session |
| `Shortname`                    | Country short code |
| `Circuit_name`                 | Name of the race circuit |
| `Lap_Time_Seconds`             | ⬅️ **Target column** |
| `Corners_per_Lap`              | Number of corners in the track |
| `Tire_Degradation_Factor_per_Lap`| Avg tire degradation per lap |
| `Pit_Stop_Duration_Seconds`    | Time spent in pit stops |
| `Ambient_Temperature_Celsius`  | Air temp |
| `Track_Temperature_Celsius`    | Track temp |
| `Weather`                      | Weather conditions |
| `Track`, `Air`, `Ground`       | Additional temp readings |
| `Starts`, `Finishes`, `With_points` | Rider career stats |
| `Podiums`, `Wins`              | Number of podiums and wins |
| `Min_year`, `Max_year`, `Years_active` | Rider experience |

</details>

---

## 📈 Feature Importance (LightGBM)

Below are the most important features learned by the model:

![Top 15 Feature Importance](https://i.postimg.cc/D0CgxCPh/Figure-1.png)
