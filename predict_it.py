import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv('petrol_consumption.csv')
var = ['Petrol_tax', 'Average_income',
       'Paved_Highways',
       'Population_Driver_licence(%)']

# for i in var:
#     plt.figure()
#     sns.regplot(x=i, y='Petrol_Consumption', data=data).set(title=f'Regression plot of {i} and Petrol Consumption')
#
# correlations = data.corr()
# sns.heatmap(correlations, annot=True).set(title="Heatmap")
# plt.show()

y = data['Petrol_Consumption']
X = data[['Petrol_tax',
          'Average_income',
          'Paved_Highways',
          'Population_Driver_licence(%)']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)

print(reg.coef_)
print(reg.intercept_)
print(X.columns)
cof_df = pd.DataFrame(data=reg.coef_,
                      index=X.columns,
                      columns=['Coefficient value'])
print(cof_df)

y_pred = reg.predict(X_test)
result = pd.DataFrame({'Actual' : y_test, 'Predicted' : y_pred})
print(result)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(reg.score(X_test, y_test))
print(reg.score(X_train, y_train))

