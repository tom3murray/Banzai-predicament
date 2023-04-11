import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
#from pmdarima.arima import auto_arima

# read in the two Excel files
whiskey = pd.read_excel('Irish Whiskey Sales by Volume.xlsx')

class_df = pd.read_excel('CLASS.xlsx')

# perform inner join on the "Category" column
merged_df = pd.merge(whiskey, class_df, on='Country', how='inner')

# Create dataframes based on the 'Quality' column values
total_df = merged_df.groupby(["Country", "Year", "Region", "Income"])["Cases"].sum().reset_index(name="Total Cases")
grouped = total_df.groupby('Country')
#standard_df = merged_df[merged_df['Quality'] == 'Standard']
#premium_df = merged_df[merged_df['Quality'] == 'Premium']
#superpremium_df = merged_df[merged_df['Quality'] == 'Super Premium']

# Create dataframes grouped by region
regions_total = total_df.groupby(["Region", "Year"])["Total Cases"].median()
regions_total = regions_total.to_frame().reset_index()
#regions_standard = standard_df.groupby(["Region", "Year"])["Cases"].median()
#regions_premium = premium_df.groupby(["Region", "Year"])["Cases"].median()
#regions_superpremium = superpremium_df.groupby(["Region", "Year"])["Cases"].median()

# Create dataframes grouped by income
incomes_total = total_df.groupby(["Income", "Year"])["Total Cases"].median()
incomes_total = incomes_total.to_frame().reset_index()
#incomes_standard = standard_df.groupby(["Income", "Year"])["Cases"].median()
#incomes_premium = premium_df.groupby(["Income", "Year"])["Cases"].median()
#incomes_superpremium = superpremium_df.groupby(["Income", "Year"])["Cases"].median()


for country, group in grouped:
    plt.plot(group['Year'], group['Total Cases'], label=country)

# Set the title and labels for the plot
plt.title('Total Irish Whiskey Imported by Country')
plt.xlabel('Year')
plt.ylabel('Total Cases')

# Add a legend to the plot
plt.legend()
# Show the plot
plt.show()

regions_grouped = regions_total.groupby('Region')
incomes_grouped = incomes_total.groupby('Income')

for region, group in regions_grouped:
    plt.plot(group['Year'], group['Total Cases'], label=region)
    
# Set the title and labels for the plot
plt.title('Total Irish Whiskey Imported by Country')
plt.xlabel('Year')
plt.ylabel('Total Cases')

# Add a legend to the plot
plt.legend()
# Show the plot
plt.show()


for income, group in incomes_grouped:
    plt.plot(group['Year'], group['Total Cases'], label=income)
    
# Set the title and labels for the plot
plt.title('Total Irish Whiskey Imported by Country')
plt.xlabel('Year')
plt.ylabel('Total Cases')

# Add a legend to the plot
plt.legend()
# Show the plot
plt.show()
