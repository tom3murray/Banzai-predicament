import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pmdarima.arima import auto_arima
from math import sqrt
from sklearn.metrics import mean_squared_error

import os
#from pmdarima.arima import auto_arima


#os.chdir('C:/Users/Lyle_/Desktop/4 Yr Sem 2/DS/Banzai-predicament/project 3/')
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
# TIME SERIES GRAPH OF TOP 8 COUNTRIES 

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
# STATIONARY TESTING 

# Group the data by 'Region' and 'Year', and sum the 'Cases'
cases_by_region_year = merged_df.groupby(['Region', 'Year'])['Cases'].sum().reset_index()

# Get the unique regions
unique_regions = cases_by_region_year['Region'].unique()

# Define a function to perform the ADF test and interpret the results
def test_stationarity(timeseries):
    result = adfuller(timeseries)
    p_value = result[1]
    
    if p_value < 0.05:
        return f"Stationary (p-value: {p_value})"
    else:
        return f"Non-stationary (p-value: {p_value})"

# Test the stationarity of the time series data for each region
stationarity_results = {}
for region in unique_regions:
    region_data = cases_by_region_year[cases_by_region_year['Region'] == region]
    stationarity_results[region] = test_stationarity(region_data['Cases'])

# Print the results
for region, result in stationarity_results.items():
    print(f"{region}: {result}")
    

#%%
# AUTOCORRELATION AND DECOMPOSITION TESTING 


# Group the data by 'Region' and 'Year', and sum the 'Cases'
cases_by_region_year = merged_df.groupby(['Region', 'Year'])['Cases'].sum().reset_index()

# Get the unique regions
unique_regions = cases_by_region_year['Region'].unique()

# Define a function to perform the Ljung-Box test and interpret the results
def test_autocorrelation(timeseries, lags=None):
    result = acorr_ljungbox(timeseries, lags=lags, return_df=True)
    p_value = result['lb_pvalue'].iloc[-1]
    
    if p_value < 0.05:
        return f"Significant autocorrelation (p-value: {p_value})"
    else:
        return f"No significant autocorrelation (p-value: {p_value})"

# Test the autocorrelation and decomposition of the time series data for each region
autocorrelation_results = {}
autocorrelations = {}
for region in unique_regions:
    region_data = cases_by_region_year[cases_by_region_year['Region'] == region]
    autocorrelation_results[region] = test_autocorrelation(region_data['Cases'])
    autocorrelations[region] = acf(region_data['Cases'], nlags=10, fft=True)

    ### CODE THAT ALSO PRODUCES GRAPHS FOR DECOMPOSITION.
    ### NOTE THAT DECOMPOSITION BREAKS DOWN THE VARIATION INTO 3 PARTS: TREND, SEASONAL,
    ### AND RESIDUAL. SEASONAL AND RESIDUAL ARE FLAT LINES IN EVERY GRAPH; 
    ### NO VARIATION IN THE DATA IS ATTRIBUTABLE TO THEM.
    # Drop region column
    region_data = region_data.drop('Region', axis=1)

    # Set "Year" column as the index
    region_data = region_data.set_index('Year')

    # Perform seasonal decomposition
    decomposition = seasonal_decompose(region_data, model='additive', period=1)

    # Get the trend, seasonal and residual components
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Plot the decomposition components
    fig, ax = plt.subplots(4, 1, figsize=(10, 10))
    ax[0].plot(region_data)
    ax[0].set_ylabel('Original')
    ax[1].plot(trend)
    ax[1].set_ylabel('Trend')
    ax[2].plot(seasonal)
    ax[2].set_ylabel('Seasonal')
    ax[3].plot(residual)
    ax[3].set_ylabel('Residual')
    plt.show()

# Print the results
for region, result in autocorrelation_results.items():
    print(f"{region}: {result}")
    print(f"Autocorrelations: {autocorrelations[region]}")
    

#%%
# ARIMA Forecasting

# Group the data by 'Region' and 'Year', and sum the 'Cases'
cases_by_region_year = merged_df.groupby(['Region', 'Year'])['Cases'].sum().reset_index()

# Get the unique regions
unique_regions = cases_by_region_year['Region'].unique()

ARIMAs = {}
for region in unique_regions:

    region_data = cases_by_region_year[cases_by_region_year['Region'] == region]

    # Drop region column
    region_data = region_data.drop('Region', axis=1)

    # Set "Year" column as the index
    region_data = region_data.set_index('Year')

    region_data.index = pd.to_datetime(region_data.index, format="%Y")
    
    # Fit the ARIMA model
    arima_model = sm.tsa.ARIMA(region_data, order=(1, 1, 1)).fit()
    
    # Forecast the next 5 years
    arima_forecast = arima_model.forecast(steps=15)
    
    # Print the forecasted values for the next 5 years
    print(arima_forecast)
    
    ARIMAs[region] = arima_forecast
    

#%%
# Another ARIMA Forecasting

# Group the data by 'Region' and 'Year', and sum the 'Cases'
cases_by_region_year = merged_df.groupby(['Region', 'Year'])['Cases'].sum().reset_index()

# Get the unique regions
unique_regions = cases_by_region_year['Region'].unique()

ARIMAs_2 = {}
for region in unique_regions:

    region_data = cases_by_region_year[cases_by_region_year['Region'] == region]

    # Drop region column
    region_data = region_data.drop('Region', axis=1)
    
    # Set "Year" column as the index
    region_data = region_data.set_index('Year')

    region_data.index = pd.to_datetime(region_data.index, format="%Y")

    train = region_data[region_data.index < pd.to_datetime("2010", format='%Y')]
    test = region_data[region_data.index >= pd.to_datetime("2010", format='%Y')]
                
    plt.plot(train['Cases'], color = "black")
    plt.plot(test['Cases'], color = "red")
    
    plt.title("Train/Test split for Whiskey Data")
    plt.ylabel("Cases")
    plt.xlabel('Year')
    sns.set()
    plt.show()

    model = auto_arima(train['Cases'], trace=True, error_action='ignore', suppress_warnings=True)
    model.fit(train['Cases'])
    forecast = model.predict(n_periods=len(test))

    rms = sqrt(mean_squared_error(test['Case'],forecast))
    print("RMSE: ", rms)




