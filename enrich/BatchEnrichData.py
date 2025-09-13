import os
import sys
import json
import logging
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain.prompts import PromptTemplate

json_data_folder = 'C:\\Users\\Jessie\\Projects\\Pokedex\\Data\\pokemon_json'
model_name = 'llama3:8b'

llm = ChatOllama(
    model=model_name,
    model_kwargs={'system':'You are an expert Pokémon game guide writer. Always write factual, player-friendly summaries based on the information you were provided.'}
)

Prompt_Template  = {
    'pokedex_summary': (
        'You are the Pokédex, a highly intelligent encyclopedia of Pokémon species.\n'
        'Given the following Pokédex entries for {name}, write a cohesive paragraph summarizing the key characteristics, behavior, and any notable facts about {name}.\n'
        'Focus on patterns and notable traits.\n'
        'Mention {name} in the response. Return only the rewritten paragraph. Do not repeat the prompt or include any boilerplate intro.\n\n'
        'Pokédex Entries:\n{data}'
    ),
    'game_location_summary': (
        'You are a strategy guide writer helping players locate Pokémon in the games.\n'
        'Based on the game location data, write a paragraph explaining where and how players can find {name} in different game versions.\n'
        'Group identical locations across games when applicable, but do not skip any entry or generation.\n'
        'Mention {name} in the response. Return only the rewritten paragraph. Do not repeat the prompt or include any boilerplate intro.\n\n'
        'Game Locations:\n{data}'
    ),
    'type_effectiveness_summary': (
        'You are a competitive Pokémon strategist.\n'
        'Given the type effectiveness details for {name}, write a strategic guide on how to use this Pokémon in battles.\n'
        'Mention its strengths, weaknesses, resistances, and any notable strategies based on the data.\n'
        'Mention {name} in the response. Return only the rewritten paragraph. Do not repeat the prompt or include any boilerplate intro.\n\n'
        'Type Effectiveness:\n{data}'
    ),
    'learnset_summary': (
        'You are a Pokémon move tutor.\n'
        'Given the learnset data for {name}, summarize what kinds of moves it can learn.\n'
        'Group similar types of moves (e.g., physical, status, type-based), and highlight any standout moves or strategies.\n'
        'Do not skip entries, but you may combine redundant ones for readability.\n'
        'Mention {name} in the response. Return only the rewritten paragraph. Do not repeat the prompt or include any boilerplate intro.\n\n'
        'Learnset Data:\n{data}'
    )
}

logging.basicConfig(filename='enrich_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')


# Build chains
chains = {}
for section, template in Prompt_Template.items():
    prompt = PromptTemplate.from_template(template)
    chains[section] = prompt | llm

# Batch
def batch_enrich(filepaths, batchsize):
    file_data_map = {}
    inputs_by_section = {section:[] for section in chains}
    file_input_map = {section: [] for section in chains}
    updated_files = set()

    for i in range(len(filepaths)):
        _file = filepaths[i]
        try:
            with open(_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            file_data_map[_file] = data 
        except Exception as e:
            logging.error(f"Failed to load {_file}: {e}")
            continue 
        for section in chains:
            if section in data and f"{section}_llm" not in data:
                inputs_by_section[section].append({
                    'name':data['name'],
                    'data':json.dumps(data[section], indent=2)
                })
                file_input_map[section].append(_file)
            if len(inputs_by_section[section]) == batchsize or i == len(filepaths)-1:
                try:
                    result = list(chains[section].batch(inputs_by_section[section]))
                    for idx,res in enumerate(result):
                        target_file = file_input_map[section][idx]
                        file_data_map[target_file][f"{section}_llm"] = res.content.strip()
                        updated_files.add(target_file)
                    
                    logging.info(f"Batched section {section} for file: {file_input_map[section]}")
                except Exception as e:
                    logging.error(f"Batch failed for section {section}: {e}")

                inputs_by_section[section].clear()
                file_input_map[section].clear()
    for _path in updated_files:
        try:
            with open(_path,'w',encoding='utf-8') as f:
                json.dump(file_data_map[_path],f,indent=2,ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save {_path}: {e}")
    
def process_all_files(folder,batchsize=5):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.json')]
    batch_enrich(files,batchsize)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("❌ Usage: python enrichData.py <path_to_pokemon_json_folder> <batchsize>")
        sys.exit(1)

    folder_path = sys.argv[1]
    batchsize = int(sys.argv[2])

    if not os.path.exists(folder_path):
        print(f"❌ Error: Path '{folder_path}' does not exist.")
        sys.exit(1)

    print(f"📂 Starting enrichment on folder: {folder_path}")
    process_all_files(folder_path,batchsize)
    print("✅ Enrichment complete.")
  











# Sequential Processing

# def enrich_pokemon_data(filepath):
#     with open(filepath,'r',encoding='utf-8') as f:
#         data = json.load(f)
    
#     updated=False
#     for section,chain in chains.items():
#         pokemon_name = data['name']
#         if section not in data:
#             continue
#         try:
#             # processed data key
#             key = f"{section}_llm_summary"
#             if key in data:
#                 continue
#             input_data = json.dumps(data[section], indent=2)
#             result = chain.invoke({'name':pokemon_name, 'data':input_data})
#             data[key] = result.content.strip()
#             updated = True
#         except Exception as e:
#             logging.error(f"Failed: {pokemon_name} - {section} - {e}")
#     if updated:
#         with open(filepath, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)
#         logging.info(f"Updated: {pokemon_name}")
#     else:
#         logging.info(f"Skipped (already enriched): {pokemon_name}")

# def process_all_files(folderpath,batch_size=5):
#     files = []
#     for filename in os.listdir(folderpath):  
#         if filename.endswith('.json'):
#             files.append(os.path.join(folderpath,filename))

# process_all_files(json_data_folder)