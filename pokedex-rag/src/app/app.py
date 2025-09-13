import os
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from src.pokedex_retrieval.retriever import PokedexRetriever
from langchain_ollama import ChatOllama

load_dotenv()

with open(r"C:\Users\Jessie\Projects\Pokedex\pokedex-rag\configs\config,yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

retriever = PokedexRetriever(
    parent_store_path=CFG["paths"]["parent_store"],
    persist_dir=CFG["paths"]["chroma_dir"],
    collection_name=CFG["vector"]["collection_name"],
    embed_model=CFG["vector"]["embed_model"],
    global_token_cap=CFG["retrieval"]["global_token_cap"],
    expand_siblings=CFG["retrieval"]["expand_siblings"],
)

llm = ChatOllama(
    model=CFG["model"]["chat_model"],
    temperature=CFG["model"]["temperature"],
)

SYSTEM = (
    "You are a factual Pokédex guide. "
    "Use ONLY the provided context. Cite sources in square brackets like [1], [2]."
)

class AskRequest(BaseModel):
    query: str

app = FastAPI(title="Pokédex RAG")

@app.post("/ask")
def ask(req: AskRequest):
    pack = retriever.retrieve(
        req.query,
        k_children=CFG["retrieval"]["k_children"],
        mmr=CFG["retrieval"]["mmr"],
        fetch_k=CFG["retrieval"]["fetch_k"],
        expand_mode="auto",
    )

    # Render context & footnotes
    blocks, footnotes = [], []
    for i, item in enumerate(pack.items, start=1):
        title = f"{item.pokemon_name} (#{item.pokemon_id}) — {' > '.join(item.section_path) or item.section_type}"
        blocks.append(f"[{i}] {title}\n{item.text}\n")
        footnotes.append(f"[{i}] {item.index_key}")

    context = "\n".join(blocks)
    debug_notes = "\n".join(footnotes)

    user_prompt = (
        f"Answer the user's question strictly from CONTEXT.\n\n"
        f"QUESTION:\n{req.query}\n\n"
        f"CONTEXT\n-------\n{context}\n\n"
        f"FOOTNOTES (debug)\n------------------\n{debug_notes}\n"
    )

    resp = llm.invoke(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
    )
    return {
        "answer": resp.content,
        "citations": pack.citations,
        "chunks_used": len(pack.items),
    }


# uvicorn src.app.app:app --host 127.0.0.1 --port 8001 --reload