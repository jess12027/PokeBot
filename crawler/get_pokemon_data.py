# %%
import re
import sys
import os
import random
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
            time.sleep(3)

# info box
def get_basic_info(driver):
    info = {}
    try:
        infobox = safe_find_element(driver,By.CSS_SELECTOR,"table.roundy.infobox")
        if not infobox:
            return []
        rows = infobox.find_elements(By.TAG_NAME, "tr") 

        for row in rows:
            # Step 1: get the td
            tds = row.find_elements(By.TAG_NAME, "td") 
            if not tds:
                continue

            for td in tds:
                # Step 2: get the header from <b> or <a><b>
                try:
                    header_elem = td.find_element(By.TAG_NAME, "b")
                    header = header_elem.text.strip()
                except:
                    continue

                # Step 3: look for nested table and extract text
                try:
                    nested_table = td.find_element(By.TAG_NAME, "table")
                    value = nested_table.text.strip()
                except:
                    value = td.text.replace(header, "").strip()

                if header and value:
                    info[header] = value
    except Exception as e:
        print("Error parsing infobox:", e) 
    return info

# clean info box
def clean_info_box(raw_info):
    clean = {}

    if 'Type' in raw_info:
        clean['Type'] = raw_info['Type'].split()
    if 'Abilities' in raw_info:
        lines = raw_info['Abilities'].splitlines()
        clean['Abilities'] = []
        for l in lines:
            if l.strip():
                clean['Abilities']  += [l.replace('Hidden Ability','').strip()]
    
    if 'Gender ratio' in raw_info:
        match = re.findall(r'([\d.]+)%\s*(male|female)', raw_info['Gender ratio'])
        if match:
            gender_ratio = {k : v + '%' for v,k in match}
            clean['gender_ratio'] = gender_ratio
    
    if 'Catch rate' in raw_info:
        parts = raw_info['Catch rate'].split('(')
        if len(parts) == 2:
            clean['catch_rate'] = {
                'value' : int(parts[0].strip()),
                'percentage' : parts[1].replace(')','').strip()
            }
    if 'Egg Groups' in raw_info:
        clean['egg_groups'] = [x.strip() for x in raw_info['Egg Groups'].split(' and ')]
    
    if 'Hatch time' in raw_info:
        clean['hatch_time'] = raw_info['Hatch time']
    
    if 'Height' in raw_info:
        parts = raw_info['Height'].split()
        if len(parts) >= 2:
            clean['height'] = {
                'imperial' : parts[0] + ' ' + parts[1],
                'metric' : parts[2] + ' ' + parts[3] if len(parts) > 3 else parts[2]
            }

    if 'Base experience yield' in raw_info:
        exp = re.findall(r'\b\d+\b', raw_info['Base experience yield'])
        if exp:
            clean['base_exp_yield'] = int(exp[0])
    
    if 'Leveling rate' in raw_info:
        clean['leveling_rate'] = raw_info['Leveling rate']

    if 'Base friendship' in raw_info:
        clean['base_friendship'] = int(re.search(r'\d+', raw_info['Base friendship']).group())

    if 'Pokédex color' in raw_info:
        clean['pokedex_color'] = raw_info['Pokédex color']

    if 'EV yield' in raw_info:
        clean['ev_yield'] = raw_info['EV yield'].replace('Total: ', '').replace('\n', '; ')
    
    return clean

# get biology and evolution
def get_section_paragraphs(driver, section_id):
    try:
        span = driver.find_element(By.ID, section_id)
    except NoSuchElementException:
        print(f"Section {section_id} not found.")
        return ""

    try:
        heading = span.find_element(By.XPATH, "./ancestor::h2 | ./ancestor::h3")
        siblings = heading.find_elements(By.XPATH, "following-sibling::*")
        paragraphs = []

        for el in siblings:
            if el.tag_name in ["h2", "h3"]:
                break  # End of section
            if el.tag_name == "p":
                paragraphs.append(el.text.strip())

        return "\n".join(paragraphs)
    except Exception as e:
        print(f"Error parsing section {section_id}: {e}")
        return ""

# pokedex entry
def get_pokedex_entries(driver):
    entries = {}
    try:
        span = driver.find_element(By.ID, "Pok.C3.A9dex_entries")
        # Get all following siblings of the <span>'s heading parent
        heading = span.find_element(By.XPATH, "./ancestor::h2 | ./ancestor::h3")
        siblings = heading.find_elements(By.XPATH, "following-sibling::*")
        
        table = None
        for el in siblings:
            if el.tag_name == "table":
                table = el
                break
            if el.tag_name in ["h2", "h3"]:
                break  # in case the section has no table
        
        if not table:
            print("No Pokédex table found")
            return entries

        rows = table.find_elements(By.XPATH, ".//tr")
        last_entry = None

        for row in rows:
            try:
                game = row.find_element(By.TAG_NAME, "th").text.strip()

                tds = row.find_elements(By.TAG_NAME, "td")
                if tds:
                    desc = tds[0].text.strip()
                    last_entry = desc
                else:
                    desc = last_entry

                if game:
                    entries[game] = desc
            except:
                continue
    except Exception as e:
        print(f"Error extracting Pokédex entries: {e}")
    return entries

# locations
def get_game_locations(driver):
    locations = {}
    try:
        span = driver.find_element(By.ID, "Game_locations")
        heading = span.find_element(By.XPATH, "./ancestor::h2 | ./ancestor::h3")
        siblings = heading.find_elements(By.XPATH, "following-sibling::*")

        tables = []
        for el in siblings:
            if el.tag_name == "table":
                tables.append(el)
            if el.tag_name in ["h2", "h3"]:
                break

        for table in tables:
            rows = table.find_elements(By.XPATH, ".//tr")
            for row in rows:
                try:
                    game = row.find_element(By.TAG_NAME, "th").text.strip()
                    td_list = row.find_elements(By.TAG_NAME, "td")
                    if td_list:
                        location = td_list[0].text.strip()
                        locations[game] = location
                except:
                    continue
    except Exception as e:
        print(f"Error extracting Game Locations: {e}")
    return locations

# type effectiveness
def get_type_effectiveness(driver):
    effectiveness = {}
    try:
        span = driver.find_element(By.ID, "Type_effectiveness")
        heading = span.find_element(By.XPATH, "./ancestor::h2 | ./ancestor::h3")
        siblings = heading.find_elements(By.XPATH, "following-sibling::*")

        for el in siblings:
            if el.tag_name == "table":
                rows = el.find_elements(By.XPATH, ".//tr")
                for row in rows:
                    try:
                        ths = row.find_elements(By.TAG_NAME, "th")
                        tds = row.find_elements(By.TAG_NAME, "td")
                        if not ths or not tds:
                            continue

                        label = ths[0].text.strip()
                        span_elements = tds[0].find_elements(By.TAG_NAME, "span")

                        types = [
                            span.text.strip()
                            for span in span_elements
                            if span.value_of_css_property("display") == "inline-block" and span.text.strip()
                        ]

                        if label and types:
                            effectiveness[label] = types
                    except:
                        continue
            if el.tag_name in ["h2", "h3"]:
                break
    except Exception as e:
        print(f"Error extracting type effectiveness: {e}")
    return effectiveness

# learnset table
def get_learnset_table(driver, section_id):
    learnset = []
    try:
        span = driver.find_element(By.ID, section_id)
        heading = span.find_element(By.XPATH, "./ancestor::*[self::h2 or self::h3 or self::h4]")
        siblings = heading.find_elements(By.XPATH, "following-sibling::*")

        table = None
        for el in siblings:
            if el.tag_name == "table":
                table = el
                break
            if el.tag_name in ["h2", "h3", "h4"]:
                break

        if not table:
            print(f"No table found for section {section_id}")
            return learnset

        rows = table.find_elements(By.XPATH, ".//tr")
        for row in rows:
            tds = row.find_elements(By.TAG_NAME, "td")
            if len(tds) >= 7:
                learnset.append({
                    "level": tds[0].text.strip(),
                    "move": tds[1].text.strip(),
                    "type": tds[2].text.strip(),
                    "category": tds[3].text.strip(),
                    "power": tds[4].text.strip(),
                    "accuracy": tds[5].text.strip(),
                    "pp": tds[6].text.strip()
                })
    except Exception as e:
        print(f"Error extracting learnset table for {section_id}: {e}")
    return learnset

def extract_trivia(driver):
    trivia_list = []

    try:
        # Locate the "Trivia" heading by id
        trivia_heading = driver.find_element(By.ID, "Trivia")

        # Navigate to its parent <h2> tag and find the next sibling <ul>
        h2_element = trivia_heading.find_element(By.XPATH, "./ancestor::h2")
        ul_element = h2_element.find_element(By.XPATH, "following-sibling::ul[1]")

        # Extract all list items
        li_elements = ul_element.find_elements(By.TAG_NAME, "li")
        for li in li_elements:
            text = li.text.strip()
            if text:
                trivia_list.append(text)

    except NoSuchElementException:
        print("Trivia section not found.")

    return trivia_list

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python get_pokemon_data.py <link_file.json>")
#         sys.exit(1)
#     link_file_path = sys.argv[1]

#     def scrape_all_from_link_file(link_file):
#         with open(link_file, "r", encoding="utf-8") as f:
#             pokemon_links = json.load(f)
        
#         batch_size = 5
#         driver = init_driver()

#         for i, entry in enumerate(pokemon_links):
#             poke_id = entry["id"]
#             url = entry["link"]
#             name = entry["name"]

#             print(f"\nScraping {poke_id} - {name}")
        
#             try:
#                 driver.get(url)
#                 wait_for_page(driver)

#                 data = {
#                     "id": poke_id,
#                     "name": name,
#                     "url": url,
#                     "basic_info": clean_info_box(get_basic_info(driver)),
#                     "biography": get_section_paragraphs(driver, "Biology"),
#                     "evolution": get_section_paragraphs(driver, "Evolution"),
#                     "pokedex": get_pokedex_entries(driver),
#                     "game_location": get_game_locations(driver),
#                     "type_effectiveness": get_type_effectiveness(driver),
#                     "learnset": {
#                         "level_up": get_learnset_table(driver, "By_leveling_up"),
#                         "tm": get_learnset_table(driver, "By_TM")
#                     },
#                     "trivia": extract_trivia(driver)
#                 }

#                 os.makedirs("Data/pokemon_json", exist_ok=True)
#                 out_path = f"Data/pokemon_json/{poke_id}_{name.lower()}.json"
#                 with open(out_path, "w", encoding="utf-8") as f:
#                     json.dump(data, f, indent=2, ensure_ascii=False)
            
#             except Exception as e:
#                 print(f"Failed to scrape {poke_id} - {name}: {e}")

#             if (i + 1) % batch_size == 0:
#                 driver.quit()
#                 print(f"Restarting driver after {i+1} Pokémon...")
#                 time.sleep(3)
#                 driver = init_driver()
#         driver.quit()
#         print("All Pokemon Finished")
#     scrape_all_from_link_file(link_file_path)



# Test
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scrape_pokemon.py <pokemon_id> <pokemon_url>")
        sys.exit(1)

    poke_id = sys.argv[1]
    url = sys.argv[2]

    driver = init_driver()
    driver.get(url)
    wait_for_page(driver)

    # Get the name from the page or derive from URL
    name = url.split("/")[-1].split("_")[0]

    data = {
        "id": poke_id,
        "name": name,
        "url": url,
        "basic_info": clean_info_box(get_basic_info(driver)),
        "biography": get_section_paragraphs(driver, "Biology"),
        "evolution": get_section_paragraphs(driver, "Evolution"),
        "pokedex": get_pokedex_entries(driver),
        "game_location": get_game_locations(driver),
        "type_effectiveness": get_type_effectiveness(driver),
        "learnset": {
            "level_up": get_learnset_table(driver, "By_leveling_up"),
            "tm": get_learnset_table(driver, "By_TM/TR")
        },
        "trivia": extract_trivia(driver)
    } 
    os.makedirs("Data/pokemon_json", exist_ok=True)
    with open(f"Data/pokemon_json/{poke_id}_{name.lower()}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    driver.quit()
    print(f"Saved data for {name} (ID: {poke_id})")