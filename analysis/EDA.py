import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/realestatedata_final.csv")
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))

df.set_index('Date', inplace=True)
select_df = df[['ZipCode', 'PctChange', 'ZHVI', 'Unemp Rate', 'Crime Rate', 'Label']]
print(select_df.head())
print()
print(select_df.describe())

print('Pairplot for all NY')
all_locations = df[['Year', 'Month', 'ZHVI', 'Unemp Rate', 'Crime Rate']]
all_locations = all_locations.groupby(['Year', 'Month']).mean().reset_index()
sns.pairplot(all_locations[['ZHVI', 'Unemp Rate', 'Crime Rate']])
plt.show()
print('Pairplot for 11553')
zip11368 = df[df['ZipCode']==11553]
zip11368_select = zip11368[['ZHVI', 'Unemp Rate', 'Crime Rate']]
sns_plot = sns.pairplot(zip11368_select)
plt.show()
all_dates = df.groupby(['Year', 'Month']).mean()
print('House value time series')
all_dates['ZHVI'].plot()
plt.show()
print('Unemployment rate time series')
all_dates['Unemp Rate'].plot()
plt.show()  
all_dates = df.groupby(['Year', 'Month']).sum()
print('Crime totals time series')
all_dates['Crime Rate'].plot()
plt.show()