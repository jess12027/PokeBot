
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import json
import time



options = Options()
options.add_argument('--headless')
driver = webdriver.Chrome(options = options)

url = 'https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number'
driver.get(url)
time.sleep(5)

pokemon_list = []
tables = driver.find_elements(By.CSS_SELECTOR, "table.roundy")

for table in tables:
    rows = table.find_elements(By.TAG_NAME,'tr')[1:]
    for row in rows:
        cells = row.find_elements(By.TAG_NAME,'td')
        if len(cells) >= 3:
            id_cell = cells[0].text.strip()
            if id_cell != '':
                id = id_cell.split()[0].replace('#','')
                name_cell = cells[2]
                name = name_cell.text.strip().split()[0]
                try:
                    link = name_cell.find_element(By.TAG_NAME,'a').get_attribute('href')
                except:
                    continue
            else:
                id = prev_id
                name_cell = cells[1]
                name = name_cell.text.strip().split()[0]
                try:
                    link = name_cell.find_element(By.TAG_NAME,'a').get_attribute('href')
                except:
                    continue

            pokemon_list.append({
                'id':id,
                'name':name,
                'link':link
            })
        prev_id = id
with open('C:\\Users\\Jessie\\Projects\\Pokedex\\Data\\pokemon_links.json','w',encoding='utf-8') as f:
    json.dump(pokemon_list,f,indent = 2, ensure_ascii=False)

print(f"Saved {len(pokemon_list)} Pokémon to pokemon_index.json")
driver.quit()

