# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:07:57 2023

@author: yaobv
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_odds(year=None):

  url = f"https://www.sportsoddshistory.com/cbb-main/?y={year}-{year+1}&sa=cbb&a=nc&o=r"

  response = requests.get(url)
  soup = BeautifulSoup(response.content, 'html.parser')

  teams = []

  for row in soup.select('tr'):
      cols = row.select('td')
      if cols:
          team_name = cols[0].get_text(strip=True)
          odds = []
          for col in cols[1:]:
              print(col)
              if col['align'] == 'right':
                  print("hit")
                  odds.append(col.get_text(strip=True))
              else:
                  odds.append('')
          teams.append([team_name] + odds)

  teams = teams[:-1]

  df = pd.DataFrame.from_records(teams)

  try:
    df = df.rename(columns={df.columns[0] : 'team', df.columns[1] : 'preseason'})
    df = df.rename(columns={df.columns[-7] : 'pretourney'})

  except:
    print('error')
    return df

  return df