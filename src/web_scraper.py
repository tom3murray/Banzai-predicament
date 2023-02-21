import pandas as pd
from bs4 import BeautifulSoup
import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0',
}

r = requests.get("http://stateoftheunion.onetwothree.net/texts/index.html", headers=headers)
content = r.content  # the content on the page that we search for with BeautifulSoup
soup = BeautifulSoup(content, features="lxml")


sou_links = soup.find("div", {"id": "text"})

sou_links = sou_links.find("ul").find_all("a")

all_data = pd.DataFrame()

presidents = []
dates = []
speeches = []
for idx, link in enumerate(sou_links):

    individual_page = requests.get(
        "http://stateoftheunion.onetwothree.net/texts/" + link['href'], headers=headers)

    content2 = individual_page.content

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

all_data.to_excel('./all_name_data.xlsx', index=False)
