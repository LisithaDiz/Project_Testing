import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv('test_1.csv')

y = data['Scores'].values.reshape(-1, 1)
X = data['Hours'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)
b = reg.intercept_
w = reg.coef_
print(b, w)


def prediction(b, w, hours):
    return w * hours + b


print(prediction(b, w, 9.5))  # or
print(reg.predict([[9.5]]))

y_pred = reg.predict(X_test)
data_pred = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(data_pred)
print('Error_Values', mean_absolute_error(y_test, y_pred),
      np.sqrt(mean_squared_error(y_test, y_pred)))

data.plot(kind='scatter',
          x='Hours',
          y='Scores',
          title='Title')
plt.plot(X_test, y_pred, color='r')
plt.show()

