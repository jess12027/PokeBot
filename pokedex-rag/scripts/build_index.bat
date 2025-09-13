@echo off
python -m src.pokedex_index.createVectorStore ^
  --input .\data\pokemon_data.txt ^
  --persist-dir .\data\chroma_child ^
  --collection-name pokemon_child_idx ^
  --parent-store .\data\parent_docstore.jsonl ^
  --child-strategy title_sim ^
  --manifest .\data\index_manifest.json