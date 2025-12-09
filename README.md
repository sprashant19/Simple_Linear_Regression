Height–Weight Linear Regression Model

This project demonstrates a complete end-to-end Linear Regression workflow using Python, including Exploratory Data Analysis (EDA), data visualization, model building, performance evaluation, and prediction.
The dataset contains two numerical features — Weight (independent variable) and Height (dependent variable).

📌 Project Overview

The objective of this project is to build a simple linear regression model that predicts Height based on Weight.
The workflow includes:

✔ Reading and exploring the dataset
✔ Data visualization using Matplotlib & Seaborn
✔ Train–test split
✔ Standardization
✔ Linear Regression model training
✔ Metrics evaluation (MAE, MSE, RMSE, R², Adjusted R²)
✔ Prediction on new data
✔ Comparison using OLS (Statsmodels)

📂 Dataset

The dataset height-weight.csv contains 23 rows:

Weight	Height
45	120
58	135
...	...

Target Variable: Height
Feature: Weight

🧰 Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-Learn

Statsmodels

🔍 Exploratory Data Analysis (EDA)

Scatter plot showing correlation between Weight and Height:

plt.scatter(df['Weight'], df['Height'])
plt.xlabel('Weight')
plt.ylabel('Height')
plt.show()


Correlation matrix:

df.corr()


A strong positive correlation (~0.93) exists between Weight and Height.

📊 Visualization

Pairplot using Seaborn:

sns.pairplot(df)
plt.show()

🧪 Model Training
Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Linear Regression
regression = LinearRegression(n_jobs=-1)
regression.fit(X_train, y_train)

Model Coefficients
Coefficient (Slope): 17.298
Intercept: 156.47

📈 Performance Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

Metric	Value
MSE	114.84
MAE	9.66
RMSE	10.71
R-Squared
score = r2_score(y_test, y_pred)


R² = 73.60%

Adjusted R-Squared
1 - (1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)


Adjusted R² = 0.67

🧮 OLS Regression (Statsmodels)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())


Provides statistical interpretation including:

t-scores

p-values

confidence intervals

🔮 Prediction on New Data

Example: Predict height for weight = 72 kg

regression.predict(scaler.transform([[72]]))


Predicted Height: 155.98 cm
