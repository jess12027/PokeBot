# build_pokedex_index.py
# ===========================================================
# INDEXING-ONLY SCRIPT for your Pokédex RAG
#
# What this script does:
#   1) Parse a single big .txt using your custom “key leaf → whole H2 parent” rules.
#   2) Create child “index cards” per parent (reusing designated summaries where applicable).
#   3) Embed CHILDREN ONLY and persist a Chroma vector store.
#   4) Save a PARENT docstore (JSONL) keyed by index_key for later expansion during retrieval.
#   5) Save a MANIFEST (JSON) with basic statistics.
#
# Notes:
#   - We embed only children. Parents are stored verbatim in a docstore for later retrieval-time expansion.
#   - Child summarization strategy is configurable: "lead" (first N sentences) or "title_sim" (TF-IDF top K sentences).
#   - Uses Chroma + Ollama embeddings if available; otherwise exits with a helpful message.
#
# Usage:
#   python build_pokedex_index.py --input all_pokemon.txt --persist-dir ./chroma_child --parent-store ./parent_docstore.jsonl --child-strategy title_sim
#
# ===========================================================

import os
import re
import json
import argparse
import unicodedata
from dataclasses import dataclass

# ---------------------------- Config (defaults) ----------------------------

SPECIAL_H2_KEYS = {
    "Game Locations": "Where to Find",                    # prefix match
    "Pokédex Entry": "Pokédex Summary",
    "Pokedex Entry": "Pokedex Summary",                   # ASCII fallback
    "Type Effectiveness": "Battle Strategy",
    "Learnset Summary": "Overview of Learnable Moves",  
}

CAPTURE_H2_INTRO = True             # keep intro text under H2 (before first H3) inside the whole-H2 parent
CHILD_ABSTRACT_SENTENCES = 3        # used by "lead"
TITLE_SIM_TOP_K_SENTENCES = 3       # used by "title_sim"

# -------------------- Vector + Embeddings ---------------------

try:
    from langchain_community.vectorstores import Chroma
    HAVE_CHROMA = True
except Exception:
    HAVE_CHROMA = False

try:
    from langchain_ollama import OllamaEmbeddings
    HAVE_OLLAMA_EMB = True
except Exception:
    HAVE_OLLAMA_EMB = False

# ------------------------ Data model ---------------------------

@dataclass
class Doc:
    page_content: str
    metadata: dict

# ------------------------- Helpers -----------------------------

def slugify(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text

def section_type_from_path(path_tuple):
    p = [t.lower() for t in path_tuple]
    if p[:1] == ["basic information"]:
        return "basic.info"
    if p[:1] == ["biography"]:
        return "bio"
    if p[:1] == ["evolution"]:
        return "evolution"
    if p[:1] == ["pokédex entry"] or p[:1] == ["pokedex entry"]:
        return "pokedex"
    if p[:1] == ["game locations"]:
        return "locations"
    if p[:1] == ["type effectiveness"]:
        return "type"
    if p[:1] == ["learnset summary"]:
        return "learnset"
    if p[:1] == ["pokémon origin"] or p[:1] == ["pokemon origin"]:
        return "additional.origin"
    if p[:1] == ["name origin"]:
        return "additional.name_origin"
    if p[:1] == ["trivia"]:
        return "additional.trivia"
    return "other"

def is_title_match(title, desired):
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii").strip().lower()
    d = unicodedata.normalize("NFKD", desired).encode("ascii", "ignore").decode("ascii").strip().lower()
    if t == d:
        return True
    if t.startswith(d):
        return True
    return d in t

def first_n_sentences(text, n):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(parts[:n]).strip()

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

def top_k_by_title_similarity(text, title_text, k):
    # TF-IDF cosine similarity (with graceful fallback to term overlap if sklearn not installed)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        title_terms = set(re.findall(r"[a-zA-Z]{3,}", title_text.lower()))
        sents = split_sentences(text)
        scored = []
        for s in sents:
            terms = set(re.findall(r"[a-zA-Z]{3,}", s.lower()))
            inter = len(terms & title_terms)
            scored.append((inter, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:k]]

    sents = split_sentences(text)
    if not sents:
        return []
    corpus = [title_text] + sents
    tfidf = TfidfVectorizer().fit_transform(corpus)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    pairs = list(zip(sims, sents))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in pairs[:k]]

# ------------------ Parsing: parents build ----------------------

def build_parents_from_marked_text(big_text):
    """
    Build parent documents per your special rules:
      - For SPECIAL_H2_KEYS, parent is the entire H2 block (intro + all H3 bodies).
      - For other H2s with no H3s, parent = H2.
      - For other H2s with H3s, each H3 = parent (leaf).
    """
    parents = []

    h1_re = re.compile(r"^#\s*Pok[eé]mon ID:\s*(\d+)\s*-\s*(.+)$", re.MULTILINE)
    h2_re = re.compile(r"^\s*##\s+(.+?)\s*$", re.MULTILINE)
    h3_re = re.compile(r"^\s*###\s+(.+?)\s*$", re.MULTILINE)

    def iter_h1_headers(big_text):
        matches = list(h1_re.finditer(big_text))
        for m in matches:
            poke_id = m.group(1).strip()
            poke_name = m.group(2).strip()
            yield {
                "id": poke_id,
                "name": poke_name,
                "start": m.end(),         # content for this Pokémon starts *after* the H1 line
                "match_start": m.start(), # useful to know where the H1 begins to slice the next block
            }
        return

    h1s = list(iter_h1_headers(big_text))
    if not h1s:
        lines = [ln for ln in big_text.splitlines() if "#" in ln][:5]
        hint = "\n".join(lines) if lines else "(no lines with '#')"
        raise ValueError(
            "No '# Pokémon ID: <id> - <name>' headers found.\n"
            f"First lines with '#':\n{hint}"
        )


    for i, h1 in enumerate(h1s):
        poke_name = h1["name"]
        poke_id   = h1["id"]
        start     = h1["start"]                    # content starts after the H1 line
        end       = h1s[i+1]["match_start"] if i+1 < len(h1s) else len(big_text)
        poke_block = big_text[start:end]

        h2s = list(h2_re.finditer(poke_block))
        if not h2s:
            content = poke_block.strip()
            if content:
                section_path = ("root",)
                parents.append(Doc(
                    page_content=content,
                    metadata={
                        "pokemon_id": poke_id,
                        "pokemon_name": poke_name,
                        "section_path": section_path,
                        "section_type": section_type_from_path(section_path),
                        "index_key": f"{poke_id}:{'/'.join(section_path)}",
                        "parent_kind": "root",
                        "slug": f"{poke_id}-{slugify(poke_name)}-root",
                    }
                ))
            continue

        for j, h2 in enumerate(h2s):
            h2_title = h2.group(1).strip()
            h2_start = h2.end()
            h2_end = h2s[j+1].start() if j+1 < len(h2s) else len(poke_block)
            h2_body = poke_block[h2_start:h2_end]

            h3s = list(h3_re.finditer(h2_body))

            # Case 1: H2 has NO H3 -> parent = H2
            if not h3s:
                content = h2_body.strip()
                if content:
                    section_path = (h2_title,)
                    parents.append(Doc(
                        page_content=content,
                        metadata={
                            "pokemon_id": poke_id,
                            "pokemon_name": poke_name,
                            "section_path": section_path,
                            "section_type": section_type_from_path(section_path),
                            "index_key": f"{poke_id}:{'/'.join(section_path)}",
                            "parent_kind": "H2_whole",
                            "slug": f"{poke_id}-{slugify(poke_name)}-{slugify(h2_title)}",
                        }
                    ))
                continue

            # Case 2: Special H2 -> parent is the ENTIRE H2
            h2_title_ascii = unicodedata.normalize("NFKD", h2_title).encode("ascii", "ignore").decode("ascii").strip()
            if h2_title in SPECIAL_H2_KEYS or h2_title_ascii in SPECIAL_H2_KEYS:
                full_h2_text = h2_body.strip()
                section_path = (h2_title,)
                parent_index_key = f"{poke_id}:{'/'.join(section_path)}"
                parents.append(Doc(
                    page_content=full_h2_text,
                    metadata={
                        "pokemon_id": poke_id,
                        "pokemon_name": poke_name,
                        "section_path": section_path,
                        "section_type": section_type_from_path(section_path),
                        "index_key": parent_index_key,
                        "parent_kind": "H2_whole_special",
                        "slug": f"{poke_id}-{slugify(poke_name)}-{slugify(h2_title)}",
                    }
                ))
                continue

            # Case 3: Non-special H2 with H3s -> each H3 is a parent (leaf)
            for k, h3 in enumerate(h3s):
                h3_title = h3.group(1).strip()
                s = h3.end()
                e = h3s[k+1].start() if k+1 < len(h3s) else len(h2_body)
                h3_body = h2_body[s:e].strip()
                if not h3_body:
                    continue

                section_path = (h2_title, h3_title)
                parents.append(Doc(
                    page_content=h3_body,
                    metadata={
                        "pokemon_id": poke_id,
                        "pokemon_name": poke_name,
                        "section_path": section_path,
                        "section_type": section_type_from_path(section_path),
                        "index_key": f"{poke_id}:{'/'.join(section_path)}",
                        "parent_kind": "H3_leaf",
                        "slug": f"{poke_id}-{slugify(poke_name)}-{slugify(h2_title)}-{slugify(h3_title)}",
                    }
                ))

    return parents

# ------------------ Child creation -----------------------------

def extract_designated_h3_text(h2_body, desired_h3_title):
    h3_re = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
    h3s = list(h3_re.finditer(h2_body))
    if not h3s:
        return None, None

    slices = []
    for k, h3 in enumerate(h3s):
        title = h3.group(1).strip()
        s = h3.end()
        e = h3s[k+1].start() if k+1 < len(h3s) else len(h2_body)
        body = h2_body[s:e].strip()
        slices.append((title, body))

    for title, body in slices:
        if is_title_match(title, desired_h3_title):
            return title, body
    return None, None

def child_text_from_lead(parent_doc):
    return first_n_sentences(parent_doc.page_content, CHILD_ABSTRACT_SENTENCES)

def top_sentences_title_sim(parent_doc):
    meta = parent_doc.metadata
    path = meta.get("section_path", [])
    title_text = " - ".join(path) if path else meta.get("section_type", "section")
    top = top_k_by_title_similarity(parent_doc.page_content, title_text, TITLE_SIM_TOP_K_SENTENCES)
    if not top:
        return child_text_from_lead(parent_doc)
    return " ".join(top)

def make_child_from_parent(parent_doc, child_strategy):
    """
    - For special H2 parents: extract child text from the designated H3 inside the H2 body.
    - For H2 without H3: generate short abstract via chosen strategy.
    - For H3 leaves: reuse content (or you can also compress via strategy if you prefer).
    """
    kind = parent_doc.metadata.get("parent_kind")

    if kind == "H2_whole_special":
        # Extract the designated H3 text from within this H2
        h2_title = parent_doc.metadata["section_path"][0]
        desired = SPECIAL_H2_KEYS.get(h2_title) or SPECIAL_H2_KEYS.get(
            unicodedata.normalize("NFKD", h2_title).encode("ascii", "ignore").decode("ascii").strip()
        )
        # We only have the whole H2 text, so we search inside it for ### blocks
        # If we can’t find the desired block, fall back to a short abstract
        title, body = extract_designated_h3_text(parent_doc.page_content, desired)
        child_text = body if body else top_sentences_title_sim(parent_doc)

    elif kind == "H2_whole":
        child_text = top_sentences_title_sim(parent_doc) if child_strategy == "title_sim" else child_text_from_lead(parent_doc)

    elif kind == "H3_leaf":
        # Usually already concise enough; if you want to compress, swap the next line:
        child_text = top_sentences_title_sim(parent_doc) if child_strategy == "title_sim" else child_text_from_lead(parent_doc)

    else:
        child_text = top_sentences_title_sim(parent_doc) if child_strategy == "title_sim" else child_text_from_lead(parent_doc)

    child = Doc(
        page_content=child_text.strip(),
        metadata={**parent_doc.metadata, "is_child": True}
    )
    return child

def create_children(parents, child_strategy):
    children = []
    for p in parents:
        children.append(make_child_from_parent(p, child_strategy))
    return children

# ------------------- Parent docstore ---------------------------

def save_parent_docstore(parents, path):
    with open(path, "w", encoding="utf-8") as f:
        for p in parents:
            rec = {
                "index_key": p.metadata["index_key"],
                "page_content": p.page_content,
                "metadata": p.metadata
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ------------------- Child vector store ------------------------
def _sanitize_metadata_for_chroma(md):
    out = {}
    for k, v in md.items():
        if k == "section_path":
            # convert tuple/list -> "H2 > H3" string, and also expose h2/h3 fields
            if isinstance(v, (list, tuple)):
                parts = [str(x) for x in v]
                out["section_path"] = " > ".join(parts)
                if len(parts) > 0:
                    out["h2"] = parts[0]
                if len(parts) > 1:
                    out["h3"] = parts[1]
            else:
                out["section_path"] = str(v)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            # fallback: stringify anything else (e.g., sets, dicts)
            out[k] = str(v)
    return out

def build_and_persist_child_chroma(children, persist_dir, collection_name, batch_size=1024):
    if not HAVE_CHROMA or not HAVE_OLLAMA_EMB:
        raise RuntimeError(
            "Chroma and/or Ollama embeddings not available. "
            "Install with: pip install chromadb langchain-community langchain-ollama"
        )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    texts = [d.page_content for d in children]
    metas = [_sanitize_metadata_for_chroma(d.metadata) for d in children]

    n = len(texts)
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        vs.add_texts(texts=texts[i:j], metadatas=metas[i:j])
        i = j

    vs.persist()
    return len(children)

# ------------------------- Manifest ----------------------------

def write_manifest(path, parents, children, args):
    meta = {
        "input_path": args.input,
        "persist_dir": args.persist_dir,
        "collection_name": args.collection_name,
        "parent_store": args.parent_store,
        "child_strategy": args.child_strategy,
        "counts": {
            "parents": len(parents),
            "children": len(children)
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# ------------------------- Main -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build parent–child Pokédex index (indexing only).")
    parser.add_argument("--input", required=True, help="Path to the big combined .txt")
    parser.add_argument("--persist-dir", default="./chroma_child", help="Chroma persist directory for CHILD index")
    parser.add_argument("--collection-name", default="pokemon_child_idx", help="Chroma collection name")
    parser.add_argument("--parent-store", default="./parent_docstore.jsonl", help="Path to write parent docstore JSONL")
    parser.add_argument("--child-strategy", choices=["lead", "title_sim"], default="title_sim", help="Child summarization method")
    parser.add_argument("--manifest", default="./index_manifest.json", help="Where to write an index manifest JSON")
    args = parser.parse_args()

    # 0) Read input file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    big_text = open(args.input, "r", encoding="utf-8").read()
    if not big_text.strip():
        raise ValueError("Input file is empty.")

    # 1) Parse parents
    parents = build_parents_from_marked_text(big_text)

    # 2) Save parent docstore
    save_parent_docstore(parents, args.parent_store)

    # 3) Build children
    children = create_children(parents, args.child_strategy)

    # 4) Build & persist CHILD vector store
    os.makedirs(args.persist_dir, exist_ok=True)
    added = build_and_persist_child_chroma(children, args.persist_dir, args.collection_name)

    # 5) Manifest
    write_manifest(args.manifest, parents, children, args)

    # 6) Done
    print(f"[OK] Parents saved: {args.parent_store} (count={len(parents)})")
    print(f"[OK] Children embedded in Chroma: dir={args.persist_dir}, collection='{args.collection_name}', added={added}")
    print(f"[OK] Manifest: {args.manifest}")

if __name__ == "__main__":
    main()
