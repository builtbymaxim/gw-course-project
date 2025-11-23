"""
Lightweight console chatbot using a small Hugging Face transformer (default: Phi-3-mini-4k-instruct).

Run:
  python simple_chatbot.py --model-id microsoft/Phi-3-mini-4k-instruct

Notes:
- The first run will download the model unless it is already cached; use --local-files-only to stay offline.
- Type 'exit' or 'quit' to end the chat.
"""

from __future__ import annotations

import argparse
from collections import deque
from typing import Deque, List, Tuple
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


SYSTEM_MESSAGE = "You are a helpful, concise assistant. Keep replies short and conversational."


def select_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_prompt(history: Deque[Tuple[str, str]], user_message: str) -> str:
    lines = [SYSTEM_MESSAGE]
    for past_user, past_bot in history:
        lines.append(f"User: {past_user}")
        lines.append(f"Bot: {past_bot}")
    lines.append(f"User: {user_message}")
    lines.append("Bot:")
    return "\n".join(lines)


def build_messages(history: Deque[Tuple[str, str]], user_message: str) -> List[dict]:
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    for past_user, past_bot in history:
        messages.append({"role": "user", "content": past_user})
        messages.append({"role": "assistant", "content": past_bot})
    messages.append({"role": "user", "content": user_message})
    return messages


def generate_reply(
    model,
    tokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def _generate():
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
                use_cache=True,
            )

    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()

    pieces: List[str] = []
    for text in streamer:
        print(text, end="", flush=True)
        pieces.append(text)

    thread.join()
    print()
    reply = "".join(pieces).strip()
    return reply or "(no response generated; try asking again.)"


def chat(args) -> None:
    device = select_device(args.device)
    print(f"Loading model '{args.model_id}' on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    model.to(device)
    model.eval()

    history: Deque[Tuple[str, str]] = deque(maxlen=args.history)
    print("Chat ready. Ask me anything or type 'exit' to quit.")

    try:
        while True:
            user_message = input("\nYou: ").strip()
            if user_message.lower() in {"exit", "quit"}:
                print("Bye.")
                break
            if not user_message:
                continue

            messages = build_messages(history, user_message)
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = build_prompt(history, user_message)
            print("Bot: ", end="", flush=True)
            reply = generate_reply(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            history.append((user_message, reply))
    except KeyboardInterrupt:
        print("\nBye.")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple console chatbot using a Hugging Face causal language model.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Model to load from Hugging Face Hub or local cache.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use cached model files (no downloads).",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=6,
        help="How many past turns to keep in the prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=120,
        help="Maximum tokens generated per reply.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (higher = more random).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling probability mass.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (e.g., cpu, cuda, mps). Defaults to auto-detect.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    chat(args)
