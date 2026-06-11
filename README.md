# Gridlock Hackathon 2.0 – Traffic Demand Prediction

## Overview

This project was developed as part of Gridlock Hackathon 2.0. The objective was to predict traffic demand using road characteristics, location information, weather conditions, and temporal features.

The solution uses CatBoost Regressor with feature engineering and categorical feature handling to model traffic demand patterns.

## Dataset Features

* geohash
* day
* timestamp
* RoadType
* NumberofLanes
* LargeVehicles
* Landmarks
* Temperature
* Weather

### Target Variable

* demand

## Data Preprocessing

### Missing Value Handling

* RoadType → "Unknown"
* Weather → "Unknown"
* Temperature → Median value

### Time Features

Extracted from timestamp:

* hour
* minute

### Engineered Features

#### Cyclical Time Features

* hour_sin
* hour_cos

These features help the model understand the cyclic nature of time.

#### Interaction Features

* road_weather
* temp_lane

## Model

### Algorithm

CatBoost Regressor

### Configuration

* Iterations: 1200
* Depth: 8
* Learning Rate: 0.05
* Loss Function: RMSE

## Results

### Validation Performance

* Validation R²: 0.941846588460324

### Leaderboard Score

* 90.28672

## Feature Importance

Top contributing features:

1. RoadType
2. geohash
3. NumberofLanes
4. LargeVehicles
5. road_weather
6. hour_sin
7. hour_cos

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* CatBoost
* Google Colab

## Repository Structure

```text
Gridlock-Hackathon-2.0/
│
├── Gridlock.ipynb
└── README.md
```


