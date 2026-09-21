"""Deterministic workload trace generation for the kvRouteRS benchmark.

Three shapes, all reproducible from a fixed seed:

- ``shared_prefix``: a long system prompt shared by many requests with short
  unique suffixes (the RAG / agent-system-prompt case; cache-aware routing
  should win big).
- ``multi_turn``: simulated conversations where turn N's prompt contains all of
  turns 1..N-1 (the chat case).
- ``random``: no shared prefixes (the control; cache-aware routing should show
  no win, and that must be reported honestly).
"""

import math
import random

VOCAB = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega token cache prefix "
    "router worker replica latency throughput prefill decode kv attention "
    "transformer embedding context window serving batch stream pipeline model "
    "retrieval document query answer system prompt history turn dialogue "
).split()


def _words(rng, count):
    return " ".join(rng.choice(VOCAB) for _ in range(count))


def canonical_chars(messages):
    """Character count of router-core's canonical prompt serialization."""
    parts = [f"{m['role']}: {m['content']}" for m in messages]
    return sum(len(part) for part in parts) + (len(parts) - 1)


def prompt_blocks(messages, block_chars=512):
    return max(1, math.ceil(canonical_chars(messages) / block_chars))


class TraceGenerator:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def _text(self, target_chars):
        words = _words(self.rng, max(8, target_chars // 7))
        return words[:target_chars]

    def _text_exact(self, target_chars):
        """Random text of *exactly* `target_chars` characters. Exact lengths
        matter: the prefix cache is block-based, so a shared prefix must end on
        a 512-char block boundary to keep the per-request suffix out of the
        shared blocks."""
        parts = []
        total = 0
        while total < target_chars:
            word = self.rng.choice(VOCAB)
            parts.append(word)
            total += len(word) + 1
        return " ".join(parts)[:target_chars]

    def shared_prefix(self, requests, prefixes, prefix_chars, suffix_chars):
        """`prefixes` distinct long system prompts, each reused by many
        requests with short unique suffixes. Prompt ordering is round-robin
        across prefixes so repeats are interleaved. The system prompt is built
        to an exact character count so the suffix always starts a fresh block."""
        system_prompts = []
        for i in range(prefixes):
            label = f"System document {i}: "
            body = self._text_exact(prefix_chars - len(label))
            system_prompts.append(label + body)
        trace = []
        for k in range(requests):
            i = k % prefixes
            suffix = self._text_exact(suffix_chars)
            messages = [
                {"role": "system", "content": system_prompts[i]},
                {"role": "user", "content": f"Question about document {i}: {suffix}"},
            ]
            trace.append({
                "messages": messages,
                "blocks": prompt_blocks(messages),
                "prefix_id": i,
            })
        return trace

    def multi_turn(self, requests, conversations, turn_chars):
        """`conversations` interleaved dialogues; turn N includes turns 1..N-1
        of the same conversation, so each turn is a prefix of the next."""
        turns_per_conversation = max(1, requests // conversations)
        conversations_data = []
        for c in range(conversations):
            turns = []
            for t in range(turns_per_conversation):
                user = f"Turn {t} question in conversation {c}: " + self._text(turn_chars // 2)
                assistant = f"Answer {t} in conversation {c}: " + self._text(turn_chars // 2)
                turns.append((user, assistant))
            conversations_data.append(turns)
        trace = []
        for k in range(requests):
            c = k % conversations
            t = k // conversations
            messages = []
            for (user, assistant) in conversations_data[c][:t]:
                messages.append({"role": "user", "content": user})
                messages.append({"role": "assistant", "content": assistant})
            user, _ = conversations_data[c][t]
            messages.append({"role": "user", "content": user})
            trace.append({
                "messages": messages,
                "blocks": prompt_blocks(messages),
                "conversation": c,
                "turn": t,
            })
        return trace

    def random(self, requests, prompt_chars):
        """Fully random prompts with no shared prefixes."""
        trace = []
        for _ in range(requests):
            messages = [{"role": "user", "content": self._text(prompt_chars)}]
            trace.append({
                "messages": messages,
                "blocks": prompt_blocks(messages),
            })
        return trace
