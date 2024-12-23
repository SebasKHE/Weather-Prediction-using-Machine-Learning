
# **Weather Prediction using Machine Learning**

## **Project Description**
This project focuses on predicting weather conditions using machine learning models applied to a climate dataset. The goal is to predict the probability of rain on the following day based on several meteorological factors. We explore and compare the performance of different machine learning models:

1. **Support Vector Machine (SVM)**
2. **K-Nearest Neighbors (KNN)**
3. **Decision Tree**
4. **Logistic Regression**

The dataset includes daily meteorological features such as temperature, rainfall, humidity, wind speed, and pressure to predict the occurrence of rain the next day.

## **Dataset**

The dataset contains daily weather data with various features. Some key features include:

| Feature Name         | Description                                                   |
|----------------------|---------------------------------------------------------------|
| `Date`               | Date of the weather record                                    |
| `MinTemp`            | Minimum temperature recorded (°C)                              |
| `MaxTemp`            | Maximum temperature recorded (°C)                              |
| `Rainfall`           | Amount of rainfall (mm)                                       |
| `Evaporation`        | Evaporation rate (mm)                                         |
| `Sunshine`           | Hours of sunshine                                            |
| `WindGustDir`        | Direction of the strongest wind gust                          |
| `WindGustSpeed`      | Speed of the strongest wind gust (km/h)                        |
| `WindDir9am`         | Wind direction at 9am                                         |
| `WindDir3pm`         | Wind direction at 3pm                                         |
| `Humidity9am`        | Humidity at 9am (%)                                           |
| `Humidity3pm`        | Humidity at 3pm (%)                                           |
| `Pressure9am`        | Pressure at 9am (hPa)                                         |
| `Pressure3pm`        | Pressure at 3pm (hPa)                                         |
| `Cloud9am`           | Cloud cover at 9am (oktas)                                    |
| `Cloud3pm`           | Cloud cover at 3pm (oktas)                                    |
| `Temp9am`            | Temperature at 9am (°C)                                       |
| `Temp3pm`            | Temperature at 3pm (°C)                                       |
| `RainToday`          | Whether it rained today (Yes/No)                              |
| `RainTomorrow`       | Whether it will rain tomorrow (Yes/No) (target variable)      |

### **Dataset Sample**

| Date       | MinTemp | MaxTemp | Rainfall | Evaporation | Sunshine | WindGustDir | WindGustSpeed | WindDir9am | WindDir3pm | Humidity9am | Humidity3pm | Pressure9am | Pressure3pm | Cloud9am | Cloud3pm | Temp9am | Temp3pm | RainToday | RainTomorrow |
|------------|---------|---------|----------|-------------|----------|-------------|---------------|------------|------------|-------------|-------------|-------------|-------------|----------|----------|---------|---------|-----------|--------------|
| 2/1/2008   | 19.5    | 22.4    | 15.6     | 6.2         | 0.0      | W           | 41            | S          | SSW        | 92          | 84          | 1017.6      | 1017.4      | 8        | 8        | 20.7    | 20.9    | Yes       | Yes          |
| 2/2/2008   | 19.5    | 25.6    | 6.0      | 3.4         | 2.7      | W           | 41            | W          | E          | 83          | 73          | 1017.9      | 1016.4      | 7        | 7        | 22.4    | 24.8    | Yes       | Yes          |
| 2/3/2008   | 21.6    | 24.5    | 6.6      | 2.4         | 0.1      | W           | 41            | ESE        | ESE        | 88          | 86          | 1016.7      | 1015.6      | 7        | 8        | 23.5    | 23.0    | Yes       | Yes          |
| 2/4/2008   | 20.2    | 22.8    | 18.8     | 2.2         | 0.0      | W           | 41            | NNE        | E          | 83          | 90          | 1014.2      | 1011.8      | 8        | 8        | 21.4    | 20.9    | Yes       | Yes          |
| 2/5/2008   | 19.7    | 25.7    | 77.4     | 4.8         | 0.0      | W           | 41            | NNE        | W          | 88          | 74          | 1008.3      | 1004.8      | 8        | 8        | 22.5    | 25.5    | Yes       | Yes          |

## **Models Compared**

We train and evaluate four different models to predict if it will rain tomorrow:

1. **Support Vector Machine (SVM)**  
   - SVM is used for classification by constructing hyperplanes in a high-dimensional space to separate different classes. 

2. **K-Nearest Neighbors (KNN)**  
   - KNN classifies data points based on the majority class of their nearest neighbors.

3. **Decision Tree**  
   - Decision trees split the data into subsets based on feature values, forming a tree-like structure to make predictions.

4. **Logistic Regression**  
   - Logistic regression models the probability that a given input belongs to a certain class, used here to predict the likelihood of rain.

## **Model Evaluation**

Here are the evaluation results for each model:

| Model               | Accuracy Score | Jaccard Score | F1 Score  | Log Loss  |
|---------------------|----------------|---------------|-----------|-----------|
| Support Vector Machine (SVM)    | 0.722137       | 0.000000      | 0.000000  | NaN       |
| K-Nearest Neighbors (KNN)      | 0.818321       | 0.425121      | 0.596610  | NaN       |
| Decision Tree               | 0.856489       | 0.520408      | 0.684564  | NaN       |
| Logistic Regression        | 0.836641       | 0.509174      | 0.674772  | 0.380451  |

### **Metrics Explained:**
- **Accuracy Score**: The proportion of correctly predicted instances.
- **Jaccard Score**: The ratio of the intersection of predicted and actual labels to the union of predicted and actual labels.
- **F1 Score**: The weighted average of precision and recall, balancing the trade-off between the two.
- **Log Loss**: A measure of the uncertainty in the predictions (only available for Logistic Regression).

## **Technologies and Tools Used**
- **Programming Language:** Python  
- **Libraries:**  
  - `scikit-learn` for model training and evaluation.  
  - `pandas` for data manipulation and preprocessing.  
  - `matplotlib` and `seaborn` for data visualization.  
- **Development Environment:** Jupyter Notebook  

## **How to Use the Project**

1. Clone this repository:  
   ```bash
   git clone https://github.com/SebasKHE/weather-prediction-using-Machine-Learning.git
   ```
2. Install the required dependencies:  
   ```bash
   pip install scikit-learn pandas matplotlib seaborn
   ```
3. Open the Jupyter notebook:  
   ```bash
   jupyter notebook notebooks/proy_final.ipynb
   ```
4. Run the cells to train and compare the models.

## **Conclusion**

This project demonstrates how different machine learning models can be applied to weather prediction. The **Decision Tree** model performed the best in terms of accuracy and F1 score, while the **Support Vector Machine (SVM)** struggled to achieve meaningful performance metrics, particularly with the Jaccard and F1 scores.

---
