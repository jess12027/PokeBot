import os
import re
import sys
import json
import unicodedata
import logging
from collections import defaultdict
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate


log_filename = 'rewriteData.log'
logging.basicConfig(
    filename = log_filename,
    filemode = 'a',
    format = '%(asctime)s - %(levelname)s - %(message)s',
    level = logging.INFO
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

model_name = 'llama3:8b'
llm = ChatOllama(
    model=model_name,
    model_kwargs={'system':'You are an expert Pokemon game guild writer. Always write factual, player-friendly summaries based on the information you were provided.'}
)

def get_basic_info(pokemon_name, basic_info):

    template = """
    This is the basic information of pokemon {pokemon_name}.
    Rewrite it into a natural language paragraph that could be used in a game guide.
    Be exhaustive and detailed. Include all entries presented by the data. Don\'t skip entries.
    'Mention {pokemon_name} in the response.Return only the rewritten paragraph. Do not repeat the prompt or include any boilerplate intro.\n\n
    'JSON:\n{basic_info}'
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt|llm
    result = chain.invoke({'pokemon_name':pokemon_name,'basic_info':basic_info})
    return result.content.strip()


def flush_group_entries(group_entry):
    lines = ''
    for entries,name in group_entry.items():
        name_str = ', '.join(name)
        lines += f"- In {name_str}: {entries}\n"
    return lines

def group_pokedex(data):
    pokedex_entries = ''
    current_generation = None
    group_entry = defaultdict(list)
    for key,value in data.items():
        if re.findall('This Pokémon was unavailable prior to',key,re.IGNORECASE):
            pokedex_entries += f"{key}\n"
        elif re.findall('This Pokémon has no Pokédex entries',key,re.IGNORECASE):
            pokedex_entries += f"{key}\n"
        elif re.findall('Generation',key):
            if current_generation and group_entry:
                # flush previous group
                pokedex_entries += flush_group_entries(group_entry)
                group_entry.clear()
            current_generation = key
            pokedex_entries += f"\nIn {current_generation}:\n"
        elif value:
            value_clean = re.sub('\n',', ',value)
            group_entry[value_clean.strip()].append(key)
    # Flush last group
    if group_entry:
        pokedex_entries += flush_group_entries(group_entry)

    return pokedex_entries
        
def group_game_location(data):
    game_locations = ''
    current_generation = None
    group_location = defaultdict(list)
    for key,value in data.items():
        
        if re.findall('^Generation',key):
            if re.findall('This Pokémon is unavailable',key,re.IGNORECASE):
                continue
            if current_generation and group_location:
                game_locations += flush_group_entries(group_location)
                group_location.clear()
            current_generation = key 
            game_locations += f"\nIn {current_generation}:\n"   
        if current_generation == 'Side Games':
            if re.findall('This Pokémon is unavailable',key,re.IGNORECASE):
                continue
            else:
                group_location[value.strip()].append(key)
        elif value:
            if re.findall('This Pokémon is unavailable', key, re.IGNORECASE):
                continue
            if current_generation == 'Generation IX' and not re.findall('Scarlet|(The Hidden Treasure)',key,re.IGNORECASE):
                game_locations += flush_group_entries(group_location)
                group_location.clear()
                current_generation = 'Side Games'
                game_locations += f"\nIn {current_generation}:\n"  
                value_clean = re.sub('\n',', ',value)
                group_location[value_clean.strip()].append(key)
            else:
                value_clean = re.sub('\n',', ',value)
                group_location[value_clean.strip()].append(key)
    if group_location:
        game_locations += flush_group_entries(group_location)
    return game_locations


def mix_datatype(s):
    total = 0
    s = s.strip()
    num = ''
    for ch in s:
        if ch.isdigit():
            num += ch 
        else:
          if num:
              total += int(num)
              num = ''
          try:
              total += unicodedata.numeric(ch)
          except(TypeError,ValueError):
              raise ValueError(f"Unable to parse character '{ch}' in mixed fraction: '{s}'")
    if num:
        total += int(num)
    return total
                


def rewrite_type_effectiveness(data,name):
    type_effectiveness = ''
    for key,value in data.items():
        key_title = re.sub('\n',' ',key).lower()
        if len(value) == 1 and value[0] == 'None':
            continue
        type_effectiveness += f"\n{name} is {key_title}\n"
        for type in value:
            type_split = type.split(' ')
            type_name = type_split[0]
            type_damage = str(mix_datatype(re.sub('×','',type_split[1]))*100)
            if key == 'Immune to:':
                type_effectiveness += f"{type_name} type, which won't cause any damage.\n"
            else:
                type_effectiveness += f"{type_name} type, which will cause {type_damage}% damage.\n"
    return type_effectiveness

def group_learnset(data,name):
    learnsets = ''
    level_up_moves = defaultdict(list)
    for key,value in data.items():
        if key == 'level_up' and len(value) > 1:
            # 
            for m in value[1:]:
                category = f"{m['category'].lower()} effect" if m['category']=='Status' else f"{m['category'].lower()} attack"
                power = m['power'] if m['power'] != '—' else ''
                accuracy = m['accuracy'] if m['accuracy'] != '—%' else ''
                if power != '' and accuracy!= '':
                    move_details = f", with a power of {power} and {accuracy} accuracy"
                elif power == '' and accuracy != '':
                    move_details = f", with a {accuracy} accuracy"
                elif power != '' and accuracy == '':
                    move_details = f", with a power of {power}"
                else:
                    move_details = ''
                move_desc = f"{m['move'].title()}, it\'s a {category}{move_details}. This move requires {m['pp']} pp (power point).\n "
                level_up_moves[m['level']].append(move_desc)
            learnsets += f"\nBy leveling up, {name} can learn the following moves:\n"
            for k,v in level_up_moves.items():
                learnsets += f"At level {k}, {name} can learn:\n"
                for move in v:
                    learnsets += f"- {move}"
        elif len(value) > 1 and value[0]['level'] != '':
            all_tm_learnset = re.split(r'\nTM|\nTR',value[0]['level'])
            learnsets += f"\nBy using TM (Technical Machine), {name} can learn the following moves:\n"
            for n in range(1,len(value)):
                ele_0_split = all_tm_learnset[n].split(' ')
                power = ele_0_split[-3]
                accuracy = ele_0_split[-2]
                pp = ele_0_split[-1]
                move_desc = value[n]['power'].lower() + ' effect' if value[n]['power'] == 'Status' else value[n]['power'].lower() + ' attack'
                if power != '—' and accuracy != '—%':
                    move_details = f", with a power of {power} and {accuracy} accuracy"
                elif power == '—' and accuracy != '—%':
                    move_details = f", with a {accuracy} accuracy"
                elif power != '—' and accuracy == '—%':
                    move_details = f", with a power of {power}"
                else:
                    move_details = ''
                learnsets += f"- {value[n]['type']} can be learnt by using {value[n]['move']}. It\'s a {move_desc}{move_details}. This move requires {pp} pp (power point).\n"
    
    return learnsets

def process_each_pokemon(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        data = json.load(f)
    # Processing raw data
    ## read name
    pokemon_name = data['name']
    
    updated = False
    ## basic info (if not already saved)
    if 'basic_info_summary' not in data:
        try:
            basic_info = data['basic_info']
            data['basic_info_summary'] = get_basic_info(pokemon_name,basic_info)
            updated = True
        except Exception as e:
            print(f"Failed to summarize basic info in {filepath}: {e}")

    ## pokedex (py script rewrite)
    try:
        pokedex_raw_data = data['pokedex']
        data['pokedex_summary'] = group_pokedex(pokedex_raw_data)
        updated = True
    except Exception as e:
        # print(f"Failed to summarize pokedex in {filepath}: {e}")
        logging.error(f"Failed to summarize basic info in {filepath}: {e}")
    
    ## game_location (py script rewrite)
    try:
        game_location_raw_data = data['game_location']
        data['game_location_summary'] = group_game_location(game_location_raw_data)
        updated = True
    except Exception as e:
        # print(f"Failed to summarize game location in {filepath}: {e}")
        logging.error(f"Failed to summarize game location in {filepath}: {e}")

    ## type_effectiveness (py script rewrite and llm provide strategy)
    try:
        type_effectiveness_raw_data = data['type_effectiveness']
        data['type_effectiveness_summary'] = rewrite_type_effectiveness(type_effectiveness_raw_data,pokemon_name)
        updated = True
    except Exception as e:
        # print(f"Failed to summarize type effectiveness in {filepath}: {e}")
        logging.error(f"Failed to summarize type effectiveness in {filepath}: {e}")

    
    ## learnset (py script)
    try:
        learnset_raw_data = data['learnset']
        data['learnset_summary'] = group_learnset(learnset_raw_data,pokemon_name)
        updated = True
    except Exception as e:
        # print(f"Failed to summarize learnset in {filepath}: {e}")
        logging.error(f"Failed to summarize learnset in {filepath}: {e}")

    if updated:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data,f,indent=2)


    

def process_raw_data(pokemon_data_folder):

    for filename in os.listdir(pokemon_data_folder):
        if filename.endswith('.json'):
            filepath = os.path.join(pokemon_data_folder,filename)
            process_each_pokemon(filepath)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python rewriteData.py <path_to_pokemon_json_folder>")
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.exists(folder_path):
        print(f"❌ Error: Path '{folder_path}' does not exist.")
        sys.exit(1)

    print(f"📂 Starting enrichment on folder: {folder_path}")
    process_raw_data(folder_path)
    print("✅ Enrichment complete.")