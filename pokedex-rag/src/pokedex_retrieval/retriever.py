# retriever.py
import os
import json
import re
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# ---------- Data Models ----------

@dataclass
class Hit:
    score: float
    child_text: str
    child_meta: Dict[str, Any]

@dataclass
class ContextItem:
    index_key: str
    pokemon_id: str
    pokemon_name: str
    section_type: str
    section_path: List[str]
    text: str

@dataclass
class ContextPack:
    """What you hand to your LLM (plus a citation map you can render in UI)."""
    items: List[ContextItem]
    citations: Dict[str, Dict[str, str]]  # index_key -> { 'pokemon': ..., 'section': ... }

# ---------- Parent Docstore Loader ----------

class ParentDocstore:
    """
    Loads parent_docstore.jsonl and builds helpful lookups:
      - by_key: index_key -> (text, metadata)
      - by_h2:  (pokemon_id, h2_title) -> [index_key...]
      - by_pid: pokemon_id -> [index_key...]
    """
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Parent docstore not found: {path}")
        self.path = path
        self.by_key: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        self.by_h2: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.by_pid: Dict[str, List[str]] = defaultdict(list)
        self._load()

    @staticmethod
    def _as_list(section_path_value: Any) -> List[str]:
        # In your indexer, tuples serialize to JSON arrays → list when reloaded.
        if isinstance(section_path_value, list):
            return [str(x) for x in section_path_value]
        if isinstance(section_path_value, str):
            # Fallback if something saved as "H2 > H3" (shouldn't happen in parent store, but be robust)
            return [s.strip() for s in section_path_value.split(">")]
        return [str(section_path_value)]

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                key = rec["index_key"]
                text = rec["page_content"]
                meta = rec["metadata"]
                self.by_key[key] = (text, meta)

                pid = str(meta.get("pokemon_id", ""))
                spath = self._as_list(meta.get("section_path", []))
                h2 = spath[0] if spath else "root"

                self.by_pid[pid].append(key)
                self.by_h2[(pid, h2)].append(key)

# ---------- Child Vector Index Loader ----------

class ChildVectorIndex:
    def __init__(self, persist_dir: str, collection_name: str, embed_model: str = "nomic-embed-text"):
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"Chroma persist dir not found: {persist_dir}")
        self.embeddings = OllamaEmbeddings(model=embed_model)
        self.vs = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )

    def search(self, query: str, k: int = 20, mmr: bool = True, fetch_k: int = 50) -> List[Hit]:
        """Return top-k child hits with scores + metadata."""
        if mmr:
            docs = self.vs.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
            # MMR path doesn't always return scores; do a secondary similarity for scores only if desired.
            scored = self.vs.similarity_search_with_score(query, k=k)
            score_map = {d.metadata.get("index_key"): s for d, s in scored}
            hits = []
            for d in docs:
                idx = d.metadata.get("index_key")
                hits.append(Hit(score=score_map.get(idx, 0.0),
                                child_text=d.page_content,
                                child_meta=d.metadata))
            return hits
        else:
            docs_scored = self.vs.similarity_search_with_score(query, k=k)
            return [Hit(score=s, child_text=d.page_content, child_meta=d.metadata) for d, s in docs_scored]

# ---------- Utilities ----------

def _norm_list(x) -> List[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, str):
        return [s.strip() for s in x.split(">")]
    return [str(x)]

def _tokenish_len(text: str) -> int:
    # Simple & fast budget proxy: ~1 token ≈ 0.75 word; we cap by words and scale.
    return max(1, len(re.findall(r"\w+", text)))

# ---------- Retriever ----------

class PokedexRetriever:
    """
    Parent-child retrieval:
      1) vector-search over child index
      2) expand to true parent text from parent_docstore
      3) (optional) pull H2 siblings for fuller coverage (e.g., Learnset H3s)
      4) pack, cap per-section, dedupe by Pokémon
    """

    def __init__(
        self,
        parent_store_path: str,
        persist_dir: str,
        collection_name: str,
        *,
        embed_model: str = "nomic-embed-text",
        section_caps: Dict[str, int] = None,
        global_token_cap: int = 1800,
        expand_siblings: bool = True
    ):
        self.parents = ParentDocstore(parent_store_path)
        self.children = ChildVectorIndex(persist_dir, collection_name, embed_model)
        self.global_token_cap = global_token_cap
        self.expand_siblings = expand_siblings
        # default section caps (rough token-ish budgets per section)
        self.section_caps = section_caps or {
            "basic.info": 220,
            "pokedex": 280,
            "bio": 220,
            "locations": 260,
            "type": 220,
            "learnset": 420,    # tends to be long
            "evolution": 180,
            "other": 160,
            "additional.origin": 120,
            "additional.name_origin": 120,
            "additional.trivia": 160,
        }

    def _parent_from_key(self, index_key: str) -> Optional[ContextItem]:
        tup = self.parents.by_key.get(index_key)
        if not tup:
            return None
        text, meta = tup
        spath = _norm_list(meta.get("section_path", []))
        return ContextItem(
            index_key=index_key,
            pokemon_id=str(meta.get("pokemon_id", "")),
            pokemon_name=str(meta.get("pokemon_name", "")),
            section_type=str(meta.get("section_type", "other")),
            section_path=spath,
            text=text.strip()
        )

    def _expand_siblings_if_needed(self, item: ContextItem) -> List[ContextItem]:
        """
        If we landed on an H3 leaf, it’s often useful to grab its sibling H3s under the same H2.
        We detect the H2 by section_path[0].
        """
        sp = item.section_path
        if len(sp) < 2:
            return [item]  # H2 whole or root
        h2 = sp[0]
        keys = self.parents.by_h2.get((item.pokemon_id, h2), [])
        out = []
        for k in keys:
            ci = self._parent_from_key(k)
            if ci:
                out.append(ci)
        # Keep stable order but place the original item first
        out_sorted = [item] + [ci for ci in out if ci.index_key != item.index_key]
        return out_sorted

    def _cap_sections_and_tokens(self, items: List[ContextItem]) -> List[ContextItem]:
        # group by (pokemon_id, section_type) to apply per-section caps
        grouped: Dict[Tuple[str, str], List[ContextItem]] = defaultdict(list)
        for it in items:
            grouped[(it.pokemon_id, it.section_type)].append(it)

        capped_items: List[ContextItem] = []
        for (pid, stype), lst in grouped.items():
            # Stable/score-agnostic: earlier items are likely closer to the retrieved child
            cap = self.section_caps.get(stype, 160)
            budget = cap
            for it in lst:
                n = _tokenish_len(it.text)
                if n <= budget:
                    capped_items.append(it)
                    budget -= n
                else:
                    # Trim text to fit remaining budget (roughly by words)
                    if budget <= 0:
                        continue
                    words = re.findall(r"\S+\s*", it.text)
                    approx = int(budget * 0.75)  # word ≈ 1.33 tokens
                    trimmed = "".join(words[:max(1, approx)]).rstrip()
                    if trimmed:
                        capped_items.append(ContextItem(
                            index_key=it.index_key,
                            pokemon_id=it.pokemon_id,
                            pokemon_name=it.pokemon_name,
                            section_type=it.section_type,
                            section_path=it.section_path,
                            text=trimmed + " …"
                        ))
                    budget = 0

        # Enforce a global cap to avoid overruns
        total = 0
        final: List[ContextItem] = []
        for it in capped_items:
            n = _tokenish_len(it.text)
            if total + n > self.global_token_cap:
                break
            final.append(it)
            total += n
        return final

    def _dedupe_and_diversify(self, items: List[ContextItem], per_pokemon_limit: int = 4) -> List[ContextItem]:
        # Keep at most N context pieces per Pokémon to avoid one Pokémon dominating.
        counts: Dict[str, int] = defaultdict(int)
        out: List[ContextItem] = []
        seen_keys = set()
        for it in items:
            if it.index_key in seen_keys:
                continue
            if counts[it.pokemon_id] >= per_pokemon_limit:
                continue
            out.append(it)
            seen_keys.add(it.index_key)
            counts[it.pokemon_id] += 1
        return out

    def retrieve(
        self,
        query: str,
        *,
        k_children: int = 24,
        mmr: bool = True,
        fetch_k: int = 64,
        expand_mode: str = "auto"  # "auto" | "parent-only" | "parent+siblings"
    ) -> ContextPack:
        """
        1) search children
        2) expand to parent
        3) optional sibling expansion
        4) dedupe/diversify
        5) cap per-section and globally
        """
        hits = self.children.search(query, k=k_children, mmr=mmr, fetch_k=fetch_k)

        # 2) expand to parent per child hit
        parents: List[ContextItem] = []
        for h in hits:
            idx = h.child_meta.get("index_key")
            if not idx:
                # Child index uses sanitized metadata; ensure index_key survived sanitize step.
                # Your indexer keeps it as original, so this should exist.
                continue
            ci = self._parent_from_key(idx)
            if not ci:
                continue
            if expand_mode in ("parent+siblings", "auto") and self.expand_siblings:
                expanded = self._expand_siblings_if_needed(ci)
                parents.extend(expanded)
            else:
                parents.append(ci)

        if not parents:
            return ContextPack(items=[], citations={})

        # 4) diversify by Pokémon to avoid single-entity domination
        diversified = self._dedupe_and_diversify(parents, per_pokemon_limit=4)

        # 5) apply per-section and global caps
        capped = self._cap_sections_and_tokens(diversified)

        # Build citations
        cits: Dict[str, Dict[str, str]] = OrderedDict()
        for it in capped:
            cits[it.index_key] = {
                "pokemon": f"{it.pokemon_name} (#{it.pokemon_id})",
                "section": " > ".join(it.section_path) if it.section_path else it.section_type
            }

        return ContextPack(items=capped, citations=cits)
