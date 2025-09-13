# %%
import re
import sys
import os
import random
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import undetected_chromedriver as uc
import json
import time
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# %%
def init_driver():
    options = uc.ChromeOptions()
    user_agent = random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
        # Add more
    ])
    options.add_argument(f"user-agent={user_agent}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.headless = False
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("start-maximized")

    driver = uc.Chrome(options=options)
    return driver


def safe_find_element(parent, by, value):
    try:
        return parent.find_element(by, value)
    except NoSuchElementException:
        return None

def wait_for_page(driver, timeout=10):
    for _ in range(timeout):
        try:
            if driver.find_element(By.ID, "Biology"):
                return
        except:
            time.sleep(random.uniform(1.5, 3.5))

def get_origin(driver, section_id):
    try:
        span = driver.find_element(By.ID, section_id)
    except NoSuchElementException:
        print(f"Section {section_id} not found.")
        return ""

    try:
        heading = span.find_element(By.XPATH, "./ancestor::h3 | ./ancestor::h4")
        siblings = heading.find_elements(By.XPATH, "following-sibling::*")
        paragraphs = []

        for el in siblings:
            if el.tag_name in ["h3", "h4"]:
                break  # End of section
            if el.tag_name == "p":
                paragraphs.append(el.text.strip())

        return "\n".join(paragraphs)
    except Exception as e:
        print(f"Error parsing section {section_id}: {e}")
        return ""
    

def enrich_pokemon_data(pokemon_data_folder,pokemon_link_path):
    
    # read link file to scrape
    with open(pokemon_link_path, 'r', encoding='utf-8') as f:
        pokemon_links = json.load(f)
    
    # restart driver after batch_size
    batch_size = 5
    driver = init_driver()

    for i, entry in enumerate(pokemon_links[1100:1200]):
        updated=False
        poke_id = entry["id"]
        url = entry["link"]
        name = entry["name"]
        # safe_name = re.sub(r'\W+', '_', name.lower())
        file_name = f"{poke_id}_{name}.json"
        try:
            pokemon_data_path = Path(pokemon_data_folder) / file_name
            # read existing data
            with open(pokemon_data_path,'r',encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {poke_id} - {name}: Cannot read file ({e})")
            continue
        print(f"\nScraping {poke_id} - {name}")
        try:
            driver.get(url)
            wait_for_page(driver)
            data['Origin'] = get_origin(driver,'Origin')
            data['Name_Origin'] = get_origin(driver,'Name_origin')
            updated = True
            # if additional data found, update the current pokemon data file
            if updated:
                with open(pokemon_data_path,'w',encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"Failed to scrape {poke_id} - {name}: {e}")

        
        if (i+1)%batch_size==0:
            driver.quit()    
            print(f"Restarting driver after {i+1} Pokémon...")
            time.sleep(random.uniform(1.5, 3.5))
            driver = init_driver()
    driver.quit()
    print("All Pokemon Finished")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python get_additional_pokemon_data.py <existing_data_folder_path> <link_file.json>")
        sys.exit(1)
    pokemon_data_folder = sys.argv[1]
    pokemon_link_path = sys.argv[2]
    enrich_pokemon_data(pokemon_data_folder,pokemon_link_path)