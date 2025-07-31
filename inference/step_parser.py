import re
from typing import List, Optional, Tuple, Dict
import torch

def build_chat_messages(question: str,tokenizer,dataset: str, shots: Optional[List[tuple[str, str, str]]] = None,) -> str:
    system_prompt = (
        "You are an **expert mathematical‑reasoning assistant**.\n\n"
        "## Format rules\n"
        "1. Begin *every* reasoning line with the exact prefix `Step k:` where `k = 1, 2, …`. No other prefix is allowed.\n"
        "2. Show *all* intermediate calculations using standard symbols (×, ÷, ±, √).\n"
        "3. Conclude with **one** line of the form `Answer: <final numeric result>` and **stop immediately** - no explanations, no closing remarks.\n"
        "4. Each step must be concise *yet mathematically rigorous*.\n"
        "5. Avoid markdown bullet lists or narrative words such as ‘First’,  ‘Next’, ‘Finally’.\n\n"
        "Follow these rules exactly - evaluations are case- and format‑sensitive.\n"
        "Respond *only* in the specified format."
    )
    default_shots: List[tuple[str, str, str]] = [
        (
            "gsm8k, math",
            "Problem: What is the next number in the sequence 2, 4, 8, 16?",
            "Step 1: Identify the pattern – each term is multiplied by 2.\n"
            "Step 2: 16 × 2 = 32\n"
            "Answer: 32",
        ),
        (
            "gsm8k, math",
            "Problem: Solve for x: 3x + 7 = 22",
            "Step 1: Subtract 7 from both sides: 3x = 15\n"
            "Step 2: Divide by 3: x = 5\n"
            "Answer: 5",
        ),
        (
            "olympiad, omni",
            "Problem: Determine whether v₁ = [1,2] and v₂ = [3,6] are linearly independent.",
            "Step 1: Observe v₂ = 3 · v₁, so v₂ is a scalar multiple of v₁.\n"
            "Step 2: Therefore the vectors are linearly dependent.\n"
            "Answer: Dependent",
        ),
    ]

    if shots is None:
        shots = default_shots

    messages = [{"role": "system", "content": system_prompt}]
    for tag, q, a in shots:
        if dataset.lower() in tag.lower():
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": f"Problem: {question}"})
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


class StepParser:
    """Extract reasoning steps from raw LLM output – robust to messy formats."""
    _STEP_RE      = re.compile(r"^\s*Step\s*(\d+)\s*[:\-]", re.I)
    _ENUM_RE      = re.compile(r"^\s*(\d+)[.)]\s+")
    _NARRATIVE_RE = re.compile(r"^\s*(First|Second|Third|Fourth|Fifth|Next|Then|After that|Finally|Lastly)\b[,:]?", re.I)

    def _split(self, text: str) -> List[str]:
        lines = text.splitlines()
        blocks: List[str] = []
        buf: List[str] = []

        def flush():
            if buf:
                blocks.append(" ".join(buf).strip())
                buf.clear()

        for ln in lines:
            if any(p.match(ln) for p in (self._STEP_RE, self._ENUM_RE, self._NARRATIVE_RE)):
                flush()
            buf.append(ln.strip())
        flush()
        return [b for b in blocks if b]

    # public method ------------------------------------------------------
    def parse(self, text: str) -> List[str]:
        steps = self._split(text)
        if len(steps) >= 2:
            return steps
        # first fallback: coarse split on blank lines (\n\n)
        paras = re.split(r"\n\s*\n", text)
        steps = [p.strip() for p in paras if p.strip()]
        if len(steps) >= 2:
            return steps
        # second fallback: line‑by‑line
        return [ln.strip() for ln in text.splitlines() if ln.strip()]

