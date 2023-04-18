import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import os
#from pmdarima.arima import auto_arima


os.chdir('C:/Users/Lyle_/Desktop/4 Yr Sem 2/DS/Banzai-predicament/project 3/')
os.getcwd()


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
plt.title('Total Irish Whiskey Imported by Region')
plt.xlabel('Year')
plt.ylabel('Total Cases')

# Add a legend to the plot
plt.legend()
# Show the plot
plt.show()




for income, group in incomes_grouped:
    plt.plot(group['Year'], group['Total Cases'], label=income)
    
# Set the title and labels for the plot
plt.title('Total Irish Whiskey Imported by AVG Income')
plt.xlabel('Year')
plt.ylabel('Total Cases')

# Add a legend to the plot
plt.legend()
# Show the plot
plt.show()
#%%


# Aggregate total cases for each country
total_cases_by_country = merged_df.groupby('Country')['Cases'].sum().reset_index()

# Sort by total cases in descending order
sorted_cases = total_cases_by_country.sort_values('Cases', ascending=False)

# Get the top 8 countries
top_8_countries = sorted_cases.head(8)

# Filter merged_df to include only the top 8 countries
filtered_df = merged_df[merged_df['Country'].isin(top_8_countries['Country'])]

# Group by 'Country' and 'Year', then sum the 'Cases'
cases_by_year = filtered_df.groupby(['Country', 'Year'])['Cases'].sum().reset_index()

# Create a line plot
plt.figure(figsize=(12, 6))

for country in top_8_countries['Country']:
    country_data = cases_by_year[cases_by_year['Country'] == country]
    plt.plot(country_data['Year'], country_data['Cases'], label=country)

plt.xlabel('Year')
plt.ylabel('Cases')
plt.title('Cases per Year for Top 8 Countries')
plt.legend()
plt.show()

#%%







