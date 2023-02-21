import pandas as pd
import numpy as np
import datetime as dt
from dateutil.parser import parse
import re

def months_until_next_election(date):
    year = date.year
    month = date.month
    if month >= 11:
        if year % 4 == 0:
            next_election_year = year + 4
            next_election_date = dt.datetime(next_election_year, 11, 1)
            months = (next_election_date - date).days // 30
            return months
    next_election_year = (year + 3) // 4 * 4
    next_election_date = dt.datetime(next_election_year, 11, 1)
    months = (next_election_date - date).days // 30
    return months

def pres_party_house_control(row):

    if row['pres_party'] == row['party1_name']:
        row['pres_house_pct_ctrl'] = round(row['party1_seats_house']/row['num_house_seats'],2)
    elif row['pres_party'] == row['party2_name']:
        row['pres_house_pct_ctrl'] = round(row['party2_seats_house']/row['num_house_seats'],2)
    else:
        row['pres_house_pct_ctrl'] = -9999

    if row['pres_house_pct_ctrl'] > .5:
        row['pres_house_pct_ctrl_bool'] = 1
    elif 0 < row['pres_house_pct_ctrl'] <= .5:
        row['pres_house_pct_ctrl_bool'] = 0
    else:
        row['pres_house_pct_ctrl_bool'] = 'goofed bool!'
        print(row['year'])
    return row['pres_house_pct_ctrl_bool']

        
def pres_party_senate_control(row):
    if row['pres_party'] == row['party1_name']:
        row['pres_senate_pct_ctrl'] = round(row['party1_seats_senate']/row['num_senate_seats'],2)
    elif row['pres_party'] == row['party2_name']:
        row['pres_senate_pct_ctrl'] = round(row['party2_seats_house']/row['num_senate_seats'],2)
    else:
        row['pres_senate_pct_ctrl'] = -9999
        
    if row['pres_senate_pct_ctrl'] >= .5:
        row['pres_senate_pct_ctrl_bool'] = 1
    elif 0 < row['pres_senate_pct_ctrl'] < .5:
        row['pres_senate_pct_ctrl_bool'] = 0
    else:
        row['pres_senate_pct_ctrl_bool'] = 'goofed bool!'
        print(row['year'])
    return row['pres_senate_pct_ctrl_bool']


variables = pd.read_excel('all_name_data.xlsx')
presidents = pd.read_excel('presidential_variables.xlsx')
years = pd.read_excel('annual_variables.xlsx')
congresses = pd.read_excel('all_congresses.xlsx')

new_variables = presidents.merge(variables, on='name', how='right')

new_variables['date'] = new_variables['date'].apply( lambda x: dt.datetime.strptime(x,'%B %d, %Y'))

new_variables['born'] = pd.to_datetime(new_variables['born'])

new_variables['age'] = (new_variables['date'] - new_variables['born']).apply( lambda x: int((x / np.timedelta64(1, 'D'))/365))

new_variables['year'] = new_variables['date'].apply( lambda x: x.year)

new_variables = years.merge(new_variables, on='year', how='right')

new_variables['months_til_pres_election'] = new_variables['date'].apply( lambda x: months_until_next_election(x) )

# Merge the sou and congresses dataframes based on year
merged = pd.merge_asof(new_variables.sort_values('year'), congresses.sort_values('start_year'),
                       left_on='year', right_on='start_year',
                       direction='backward')

# Fill null values in the congress_num column with the last known value
merged['congress'] = merged['congress'].fillna(method='ffill')

# Re-sort the merged dataframe based on the original order of the sou dataframe
merged = merged.sort_index()


ctrl_house = merged.apply( lambda x: pres_party_house_control(x), axis=1)

ctrl_senate = merged.apply( lambda x: pres_party_senate_control(x), axis=1)

merged['pres_house_pct_ctrl_bool'] = ctrl_house

merged['pres_senate_pct_ctrl_bool'] = ctrl_senate

merged['word_count'] = merged['speech'].apply( lambda x: len(re.findall(r'\w+', x)))

full_data = merged

regression_data = full_data.drop(['year', 'name', 'term', 'born', 'died', 'age_inaug', 'age_death', 'date', 'congress', 'start_year', 'end_year', 'num_house_seats', 'party1_seats_house', 'party2_seats_house', 'party1_name', 'party2_name', 'num_senate_seats', 'party1_seats_senate', 'party2_seats_senate'], axis=1)

full_data.to_excel('./full_data.xlsx', index=False)

regression_data.to_excel('./regression_data.xlsx', index=False)
