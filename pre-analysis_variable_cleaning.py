import pandas as pd
import numpy as np
import datetime as dt
from dateutil.parser import parse


variables = pd.read_excel('all_name_data.xlsx')
presidents = pd.read_excel('presidential_variables.xlsx')
years = pd.read_excel('annual_variables.xlsx')

new_variables = presidents.merge(variables, on='name', how='right')

new_variables['date'] = new_variables['date'].apply( lambda x: dt.datetime.strptime(x,'%B %d, %Y'))

new_variables['born'] = pd.to_datetime(new_variables['born'])

new_variables['age'] = (new_variables['date'] - new_variables['born']).apply( lambda x: int((x / np.timedelta64(1, 'D'))/365))

