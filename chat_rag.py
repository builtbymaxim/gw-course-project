"""
Console RAG chat over the cached PDF embeddings with an optional local LLM judge.

- Assumes you already ran notebooks/7_RAG_Verification to create:
  ../data/processed/knowledge_base.pkl and embeddings.npy
- Uses the same encoder (all-MiniLM-L6-v2) for query encoding.
- Optionally uses a small local text-generation model to answer based on retrieved evidence.
- Keeps short chat history for more natural replies and returns NOT_FOUND when evidence is weak.

Run:
  python chat_rag.py

Optional flags:
  --k 5                      # number of evidence chunks to retrieve
  --model-id microsoft/Phi-3-mini-4k-instruct   # local judge model (transformers)
  --no-model                 # skip loading a local model, just show evidence
  --min-sim 0.3              # minimum best similarity to attempt an answer
  --history 6                # how many past turns to keep in prompt
  --show-evidence            # print evidence snippets before answering
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import deque
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
CACHE_DIR = (ROOT / "data" / "processed").resolve()
DOCS_PATH = CACHE_DIR / "knowledge_base.pkl"
VECTORS_PATH = CACHE_DIR / "embeddings.npy"
DEFAULT_ENCODER = "all-MiniLM-L6-v2"
DEFAULT_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_MIN_SIM = 0.35


def load_cache() -> tuple[List[Dict[str, Any]], np.ndarray]:
    if not DOCS_PATH.exists() or not VECTORS_PATH.exists():
        raise FileNotFoundError(
            f"Cached files not found. Expected:\n- {DOCS_PATH}\n- {VECTORS_PATH}\n"
            "Run the ingestion notebook first to build them."
        )
    with DOCS_PATH.open("rb") as f:
        docs = pickle.load(f)
    vectors = np.load(VECTORS_PATH)
    return docs, vectors


def load_encoder(name: str = DEFAULT_ENCODER) -> SentenceTransformer:
    print(f"Loading encoder: {name}")
    return SentenceTransformer(name)


def maybe_load_local_model(model_id: str | None, offline: bool = False):
    if model_id is None:
        print("Skipping local model (evidence-only mode).")
        return None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=offline)
        model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=offline)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=160,
            do_sample=False,
            temperature=1.0,
        )
        print(f"✓ Local judge loaded: {model_id} ({'offline' if offline else 'online allowed'})")
        return pipe
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Could not load local model '{model_id}': {e}")
        if offline:
            print("Hint: rerun without --offline to allow downloads if the model is not cached.")
        print("Continuing in evidence-only mode.")
        return None


def normalize_vectors(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def retrieve(question: str, encoder: SentenceTransformer, db_norm: np.ndarray, docs: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    query_vec = encoder.encode([question])
    query_vec = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-12)
    scores = np.dot(db_norm, query_vec.squeeze())
    top_idx = scores.argsort()[::-1][:k]
    results = []
    for idx in top_idx:
        chunk = docs[idx]
        results.append(
            {
                "text": chunk.get("text", ""),
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "score": float(scores[idx]),
            }
        )
    return results


def format_chat_prompt(history, question: str, evidence: List[Dict[str, Any]], max_chars: int = 400) -> str:
    hist_lines = []
    for role, msg in history:
        hist_lines.append(f"{role.upper()}: {msg}")
    history_str = "\n".join(hist_lines)
    blocks = []
    for i, item in enumerate(evidence, 1):
        txt = item["text"]
        if len(txt) > max_chars:
            txt = txt[:max_chars] + " ..."
        blocks.append(
            f"[Evidence {i} | {item['source']} p.{item['page']} | sim={item['score']:.3f}]\n{txt}"
        )
    context = "\n\n".join(blocks)
    return (
        "You are a friendly, concise assistant. Speak naturally (no bullet points). You MUST ground every answer in the evidence. "
        "If evidence is insufficient, reply NOT_FOUND (or say you don't know because it is not in the annual reports).\n"
        "Answer in 1–2 short sentences and include a short source note like 'Source: file p.#' when possible.\n"
        f"{history_str}\n"
        f"USER: {question}\n"
        f"Evidence:\n{context}\n\n"
        "ASSISTANT:"
    )


def print_evidence(evidence: List[Dict[str, Any]], snippet_chars: int = 240) -> None:
    print("\nTop evidence:")
    for i, item in enumerate(evidence, 1):
        txt = item["text"]
        if len(txt) > snippet_chars:
            txt = txt[:snippet_chars] + " ..."
        print(f"- #{i} (sim={item['score']:.3f}) {item['source']} p.{item['page']}")
        print(f'  "{txt}"')


def chat_loop(args) -> None:
    docs, vectors = load_cache()
    encoder = load_encoder()
    db_norm = normalize_vectors(vectors)
    local_model = None if args.no_model else maybe_load_local_model(args.model_id, offline=args.offline)
    history = deque(maxlen=args.history)

    print(f"\nLoaded {len(docs)} chunks. Type your question, or 'exit' to quit.")
    try:
        while True:
            user_q = input("\nYou: ").strip()
            if user_q.lower() in {"exit", "quit"}:
                print("Bye.")
                break
            if not user_q:
                continue
            if len(user_q.split()) < 3:
                print("Bot: I'm focused on the annual reports—ask me something about them (e.g., AUM, revenue, strategy).")
                continue

            evidence = retrieve(user_q, encoder, db_norm, docs, args.k)
            if not evidence:
                print("Bot: NOT_FOUND (no evidence retrieved).")
                continue

            best_score = max(e["score"] for e in evidence)
            if best_score < args.min_sim:
                print(f"Bot: I'm focused on these reports and don't see enough relevant evidence (best sim {best_score:.3f} < {args.min_sim}).")
                continue

            if args.show_evidence or local_model is None:
                print_evidence(evidence)

            if local_model is None:
                print("Bot: (no local model) Answer manually using the evidence above.")
                continue

            prompt = format_chat_prompt(history, user_q, evidence)
            try:
                resp = local_model(prompt)[0]["generated_text"].strip()
                print("\nBot:", resp)
                history.append(("user", user_q))
                history.append(("assistant", resp))
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ Local model failed: {e}")
                print("Bot: NOT_FOUND (model error).")
    except KeyboardInterrupt:
        print("\nBye.")


def parse_args():
    parser = argparse.ArgumentParser(description="Console RAG chat over PDF embeddings.")
    parser.add_argument("--k", type=int, default=5, help="Top-k evidence to retrieve.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Local transformers model id.")
    parser.add_argument("--no-model", action="store_true", help="Skip loading a local model; show evidence only.")
    parser.add_argument("--offline", action="store_true", help="Force local model load from cache only (no downloads).")
    parser.add_argument("--min-sim", type=float, default=DEFAULT_MIN_SIM, help="Minimum best similarity to attempt an answer.")
    parser.add_argument("--history", type=int, default=6, help="How many past turns to keep in prompt.")
    parser.add_argument("--show-evidence", action="store_true", help="Print evidence snippets before answering.")
    return parser.parse_args()


if __name__ == "__main__":
    if not CACHE_DIR.exists():
        print(f"Cache directory not found: {CACHE_DIR}")
        sys.exit(1)
    args = parse_args()
    chat_loop(args)
