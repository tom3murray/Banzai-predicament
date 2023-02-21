# Banzai-predicament
DS 4002 Projects

## Group information
### Rallying Cry: "Here it comes.... AHHHHHHHH"
### National Anthem: The Foggy Dew
### Patron animal: Walrus (googoogachoo)

## Repo contents
### Installing/Building our code
Our code can be installed by anyone with access to this repo and immediately run on their computer; it requires no unique adjustments. 
We built our code to scrape, transform, and analyze data obtained from the internet. In cases where data could not easily be scraped, we used still used internet resources to obtain our data (and either copied it or entered by hand). We worked hard so you don't have to; all of this data is contained in our repo for your use and the references our contained in our data dictionary.
### Usage of our code
As mentioned, our code is comptaible with use on any device containing a python environment. Just clone the repo, open up your environment, open our code, and hit "Run". We allow anyone and everyone to use and copy our code.
### src Folder Files
Contains the source code for our project.

web_scraper: this code scrapes data from the internet. Specifically, it scrapes data from [this website](http://stateoftheunion.onetwothree.net/texts/index.html) [1] to give us data on all of the state of the union addresses from 1790 to 2021.

speech_sentiment_analysis: this code analyzes the text of every state of the union speech, assigning it some float value 1-10. It then gets attached to the dataframe obtained from the web_scraper files.

pre-analysis_variable_cleaning: this code takes all of the data obtained by the web_scraper file and speech_sentiment_analysis file, as well as other data contained in the "annual_variables", "all_congresses", and "presidential_variables" excel files. It assigns data to each observation (state of the union speech). It produces a "full_data" excel file, which contains all the variables created and used throughout our process. It also creates the "regression_data" excel file, which is a subset of full_data containing only the variables necessary for the regression.

OLS_regression_code: this code executes an OLS regression on the regression_data file, trying to find if there is a correlation between whether the country is at war and the positivitiy of a state of the union speech.

### Data
File name | Description | Reference
--- | --- | ---
addresses | this file contains txt files labelled as years. Each txt file represents one state of the union speech; the title is the year it was delivered; the contents is just the speech. | [2]
all_congresses | this file has each individual sitting US congress as observations. Other variables are included such as the parties in that congress (by house) and the proportion of members therein. | [3] [4]
presidential_variables | this file has each individual US president as observations. Other variables include his party and birthdate (to calculate age at time of speech). | [5]
annual variables | this file contains each year of US history (integers) as observations. Other variables are binaries such as recession (whether we were in a recession in that year), wartime, and several modified versions of the wartime binary (excluding certain wars such as the Indian War, which was technically waged at small scale for 80 years). | [6] [7] [8] [9]
full_data | this file was produced by the analysis_variable_cleaning.py file. It contains all the variables that were used throughout the processing of our dataset, including many irrelevant to our regression. |
regression_data | this file was also produced by the analysis_variable_cleaning.py file. It is a subset of full_data containing only variables needed for the final regression analysis. | 

### Figures
Figure name | Description and importance
--- | ---
figure x | description and importance of figure x are ...


### References
[1] http://stateoftheunion.onetwothree.net/texts/index.html

[2] http://stateoftheunion.onetwothree.net/texts/index.html

[3] https://history.house.gov/Institution/Party-Divisions/Party-Divisions/

[4] https://www.senate.gov/history/partydiv.htm

[5] https://www.britannica.com/place/United-States/Presidents-of-the-United-States

[6] https://www.thebalancemoney.com/the-history-of-recessions-in-the-united-states-3306011

[7] https://fred.stlouisfed.org/series/JHDUSRGDPBR

[8] https://apnews.com/article/c7ff3b454064853dc2ca78a2ee186f5f

[9] https://www.va.gov/opa/publications/factsheets/fs_americas_wars.pdf
