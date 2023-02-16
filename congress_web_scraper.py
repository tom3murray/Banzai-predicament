import pandas as pd
from bs4 import BeautifulSoup
import requests
import re

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0',
}

r = requests.get("https://www.senate.gov/history/partydiv.htm", headers=headers)
content = r.content  # the content on the page that we search for with BeautifulSoup
soup = BeautifulSoup(content, features="lxml")

all_content = soup.find("div", {"class": "contenttext"}).find_all("p")

barrier = '-'*10

new_congress_bool = False

party1_list = ['Pro-Administration', 'Federalist', 'Jackson & Crawford Republican', 'Jacksonian','Democrat']

party2_list = ['Republican']

for idx, p_tag in enumerate(all_content):
    
    p_tag_text = p_tag.text
    #print(p_tag_text)

    if barrier in p_tag_text:
        new_congress_bool = True
        continue
    
    if new_congress_bool is True:
        num = re.findall(r"[0-9]{1,3}(?:st|nd|rd|th)", p_tag_text)     

        if num:
            num_congress = num[0]
            num_congress = re.sub('\D', '', num_congress)
        elif 'Majority Party' in p_tag_text:
            
            if num_congress < 35:
                if any(ele in p_tag_text for ele in party1_list) or 'Republican' in p_tag_text:
                    party1 = p_tag_text.split(':')[1].split('(').strip()
                    if ['Pro-Administration', 'Federalist'] in party1:
                        party1_final = 'Federalist'
                    else:
                        party1_final = 'Pre-FDR Democratic'
                elif 
            
            
        elif 'Minority Party' in p_tag_text:        
        elif 'Total Seats' in p_tag_text:
        
        
    
    new_congress_bool = False

    """
    
    soup2 = BeautifulSoup(content2, features="lxml")
    page_content = soup2.find("div", {"id": "text"})

    president = page_content.find("h2").text
    date = page_content.find("h3").text
    
    speech_paras = page_content.findAll('p')
    
    speech = ''
    for para in speech_paras:
        para_text = para.text
        speech += para_text
        
    presidents.append(president)
    dates.append(date)
    speeches.append(speech)

    all_data = all_data.append({'President': president, 'Date': date, 'Speech': speech}, ignore_index=True)

all_data.to_excel('', index=False)
"""