# DS_ASSIGNMENT
## Name: Yadnyesh Sawant 
## Prn: 1272250066
# House Price Prediction (Ames Housing Dataset)

## Dataset Download
Download dataset from:
https://www.kaggle.com/datasets/prevek18/ames-housing-dataset

Place the CSV file in the same folder as this notebook.


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
```

## Load Dataset


```python
df = pd.read_csv('AmesHousing.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Alley</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>...</th>
      <th>Pool Area</th>
      <th>Pool QC</th>
      <th>Fence</th>
      <th>Misc Feature</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>Sale Type</th>
      <th>Sale Condition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>526301100</td>
      <td>20</td>
      <td>RL</td>
      <td>141.0</td>
      <td>31770</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>526350040</td>
      <td>20</td>
      <td>RH</td>
      <td>80.0</td>
      <td>11622</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>105000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>526351010</td>
      <td>20</td>
      <td>RL</td>
      <td>81.0</td>
      <td>14267</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>172000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>526353030</td>
      <td>20</td>
      <td>RL</td>
      <td>93.0</td>
      <td>11160</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>244000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>527105010</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>13830</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>189900</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 82 columns</p>
</div>



## Data Preprocessing


```python
# Drop columns with too many missing values
df = df.dropna(axis=1, thresh=0.7*len(df))

# Fill remaining missing values
df = df.fillna(df.median(numeric_only=True))

# Convert categorical to numeric
df = pd.get_dummies(df, drop_first=True)
```

## Features & Target


```python
X = df.drop('SalePrice', axis=1)
y = np.log(df['SalePrice'])
```

## Train-Test Split


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Model 1: Linear Regression


```python
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print('Linear Regression R2:', r2_score(y_test, y_pred_lr))
print('Linear Regression MAE:', mean_absolute_error(y_test, y_pred_lr))
```

    Linear Regression R2: 0.8886333737192222
    Linear Regression MAE: 0.08122932553383795
    

## Model 2: Random Forest


```python
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print('Random Forest R2:', r2_score(y_test, y_pred_rf))
print('Random Forest MAE:', mean_absolute_error(y_test, y_pred_rf))
```

    Random Forest R2: 0.919755479880196
    Random Forest MAE: 0.0850812365937336
    

## Select Best Model


```python
best_model = rf
```

## Save Model in pickle file


```python
import pickle

model_data = {
    'model': best_model,
    'features': X.columns.tolist()
}

pickle.dump(model_data, open('ames_model.pkl', 'wb'))
```
## Streamlit ui iamges 
<img width="2559" height="1599" alt="Before Model Prediction" src="https://github.com/user-attachments/assets/25c58590-efc2-4fb7-b524-031e1eacec5e" />
<img width="2559" height="1599" alt="After Model Prediction" src="https://github.com/user-attachments/assets/aebcd153-47c9-43e6-9627-4173ff34b709" />

