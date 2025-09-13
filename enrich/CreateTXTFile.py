import os
import sys
import json

def create_txt_file(pokemon_data_folder,output_file):
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for filename in os.listdir(pokemon_data_folder):
            if not filename.endswith('.json'):
                continue  

            filepath = os.path.join(pokemon_data_folder,filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            name = data.get("name", "Unknown")
            id_ = data.get("id", "N/A")
            def get_data(key,fallback = ('(No information available.)')):
                value = data.get(key)
                if key == 'trivia' and isinstance(value, list):
                    return "\n- " + "\n- ".join(value) if value else fallback
                elif isinstance(value, str):
                    return value.strip() or fallback  
                return fallback

            content = f"""# Pokémon ID: {id_} - {name}

## Basic Information
{get_data("basic_info_summary")}

## Biography
{get_data("biography")}

## Evolution
{get_data("evolution", "No evolution information available.")}

## Pokédex Entry

### Pokédex Summary
{get_data("pokedex_summary_llm")}

### Generation-Specific Descriptions
{get_data("pokedex_summary")}

## Game Locations

### General Location Summary
{get_data("game_location_summary")}

### Where to Find {name}
{get_data("game_location_summary_llm")}

## Type Effectiveness

### Type Matchups
{get_data("type_effectiveness_summary")}

### Battle Strategy
{get_data("type_effectiveness_summary_llm")}

## Learnset Summary

### Overview of Learnable Moves
{get_data("learnset_summary_llm")}

### Move Details (Leveling Up / TM/ TR)
{get_data("learnset_summary")}

## Pokémon Origin
{get_data("Origin")}

## Name Origin
{get_data("Name_Origin")}

## Trivia
{get_data("trivia")}


============================================================
"""

            out_f.write(content)

    print(f"✅ All Pokémon data written to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("❌ Usage: python rewriteData.py <path_to_pokemon_json_folder> <to_save_txt_file>")
        sys.exit(1)

    folder_path = sys.argv[1]
    to_save_txt_path = sys.argv[2]

    if not os.path.exists(folder_path):
        print(f"❌ Error: Path '{folder_path}' does not exist.")
        sys.exit(1)


    print(f"📂 Starting creating pokemon data text file: {to_save_txt_path}")
    create_txt_file(folder_path,to_save_txt_path)
    print("✅ Txt file creation completed.")