"""A real conversational chatbot, powered by a local LLM (via Ollama) with a
live fact-lookup tool bolted on.

Why this exists: the from-scratch transformer in model.py is a genuine neural
network, but at a few million parameters it can only ever be a narrow
specialist (we proved that with the math model). General conversation, broad
knowledge, and multi-turn reasoning require a model trained at a scale no
personal machine can replicate from scratch. So instead of training one,
this wires an *already-trained* open model (Llama 3.2, running locally and
free via Ollama) up as the "brain," and gives it a tool -- ask.py's live
web/Wikipedia/World-Bank lookup -- for anything needing current, precise
facts. This is the same pattern real assistants use: a language model that
knows when to call a tool instead of guessing.

Requirements:
    - Ollama installed and running (https://ollama.com)
    - A model pulled, e.g.:  ollama pull llama3.2

Usage:
    python chat.py
    python chat.py --model llama3.2
"""

import argparse
import json
import re
import sys
import urllib.request

from ask import answer as lookup_fact

# --- Deterministic tool gate -------------------------------------------------
# A 3B model isn't reliable at deciding *when* to look something up, so we don't
# leave it to the model. We decide here, in code, whether the fact-lookup tool
# is even offered for a given message. Greetings, opinions, creative requests
# and arithmetic never get the tool; clearly factual questions do.

# Things we should never look up online, even if phrased as a question.
_NEVER_LOOKUP = (
    "favorite", "favourite", "do you think", "your opinion", "in your opinion",
    "how are you", "how's it going", "hows it going", "who are you",
    "what can you do", "write ", "haiku", "poem", "story", "joke", "rap",
    "thank", "hello", "hey ", "good morning", "good night",
)

# Strong signals that a real-world fact is being requested.
_FACT_KEYWORDS = (
    "gdp", "population", "capital", "currency", "president", "prime minister",
    "inflation", "unemployment", "exports", "imports", "life expectancy",
    "literacy", "born", "died", "founded", "invented", "discovered",
    "tallest", "largest", "highest", "longest", "deepest", "biggest",
    "distance", "height", "area", "military spending", "internet users",
)

# Factual question openers ("how many people...", "when was...", etc.).
_FACT_PATTERN = re.compile(
    r"\b(who|when|where|how many|how much|how tall|how far|how old|how big|"
    r"how long|how high|how deep|what year|what is the)\b"
)

# Arithmetic like "17 * 4" or "17 times 4" -> the model should just compute it.
_ARITH_PATTERN = re.compile(
    r"\d+\s*(?:[-+*x×/]|times|plus|minus|divided by|multiplied by)\s*\d+"
)


def needs_lookup(text: str) -> bool:
    """Decide whether the fact-lookup tool should be offered for this message."""
    t = text.lower()
    if any(phrase in t for phrase in _NEVER_LOOKUP):
        return False
    if _ARITH_PATTERN.search(t):
        return False
    if any(word in t for word in _FACT_KEYWORDS):
        return True
    return bool(_FACT_PATTERN.search(t))

OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = (
    "You are a helpful, conversational assistant. The ONLY tool you have is "
    "lookup_fact -- never invent or call any other tool/function, and never "
    "print JSON or function-call-looking text as your answer.\n\n"
    "Only call lookup_fact for real-world facts you might get wrong: current "
    "statistics (GDP, population), historical dates, or facts about a "
    "specific named person/place/thing.\n\n"
    "Do NOT call any tool for: greetings and small talk, opinions, creative "
    "writing, or math/arithmetic/logic you can work out yourself -- for "
    "those, just answer directly in plain conversational text."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_fact",
            "description": (
                "Look up a factual, current-events, or statistical answer "
                "online (World Bank stats, Wikipedia summaries, or general "
                "web abstract). Use this for things like GDP, population, "
                "historical dates, measurements, or 'who/what/when/where' "
                "questions about real people, places, or things."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The factual question to look up, in plain English.",
                    }
                },
                "required": ["question"],
            },
        },
    }
]


def _post(payload: dict) -> dict:
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _looks_broken(content: str) -> bool:
    """Small local models occasionally emit an empty or bare-JSON glitch
    instead of real text. Retrying once usually recovers a normal answer."""
    c = content.strip()
    return c in ("", "{}") or (c.startswith("{") and c.endswith("}"))


def chat_turn(model: str, messages: list) -> str:
    """Send the conversation to the model, handling one round of tool use.

    The fact-lookup tool is only offered when the gate (needs_lookup) approves
    the latest user message -- so the model literally can't over-trigger it on
    small talk, opinions, or arithmetic.
    """
    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )
    tools = TOOLS if needs_lookup(last_user) else []

    response = _post(
        {"model": model, "messages": messages, "tools": tools, "stream": False}
    )
    msg = response["message"]

    tool_calls = msg.get("tool_calls")
    if not tool_calls:
        content = msg.get("content", "")
        if _looks_broken(content):
            retry = _post(
                {"model": model, "messages": messages, "tools": tools, "stream": False}
            )
            msg = retry["message"]
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                messages.append(msg)
                return msg.get("content", "")
        else:
            messages.append(msg)
            return content

    # The model wants a fact -- run our tool locally and feed the result back.
    messages.append(msg)
    for call in tool_calls:
        args = call["function"]["arguments"]
        question = args.get("question", "")
        print(f"  [looking up: {question}]")
        try:
            result = lookup_fact(question)
        except Exception as e:
            result = f"lookup failed: {e}"
        messages.append({"role": "tool", "content": result})

    # Second pass: let the model phrase a natural answer using the tool result.
    response2 = _post(
        {"model": model, "messages": messages, "tools": tools, "stream": False}
    )
    msg2 = response2["message"]
    messages.append(msg2)
    return msg2.get("content", "")


def main():
    parser = argparse.ArgumentParser(description="Chat with a local LLM + fact lookup.")
    parser.add_argument("--model", default="oikos-chat")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Chatting with {args.model} (Ctrl+C or 'exit' to quit)\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        reply = chat_turn(args.model, messages)
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()
