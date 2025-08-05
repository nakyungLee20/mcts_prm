import re
from typing import List, Optional, Tuple, Dict, Union, Iterable
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
            "gsm8k, math, olympiad, omni",
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


def build_chat_messages2(question: str, tokenizer, dataset: str, shots: Optional[List[Tuple[str, str, str]]] = None,) -> str:
    system_prompt = (
        "Solve the math problem step by step. "
        "Only generate **one new step** based on the current prefix. "
        "If you reach the final solution, output exactly: 'Answer: [final answer]' and stop. "
        "Do not generate extra text or additional steps after the answer."
    )

    default_shots: List[Tuple[str, str, str]] = [
        (
            "gsm8k, math, olympiad, omni",
            "Problem: What is the next number in the sequence 2, 4, 8, 16?\nStep 1: Identify the pattern – each term is multiplied by 2.\n",
            "Step 2: 16 × 2 = 32\n" "Answer: 32",
        ),
        (
            "gsm8k, math, olympiad, omni",
            "Problem: Solve for x: 3x + 7 = 22\nStep 1: Subtract 7 from both sides: 3x = 15\nStep 2: Divide by 3: x = 5\n",
            "Answer: 5",
        ),
        # (
        #     "olympiad, omni",
        #     "Problem: Determine whether v₁ = [1,2] and v₂ = [3,6] are linearly independent.",
        #     "Step 1: Observe v₂ = 3 · v₁, so v₂ is a scalar multiple of v₁.\n"
        #     "Step 2: Therefore the vectors are linearly dependent.\n" "Answer: Dependent",
        # ),
    ]

    if shots is None:
        shots = default_shots

    messages = [{"role": "system", "content": system_prompt}]
    for tag, q, a in shots:
        if dataset.lower() in tag.lower():
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": f"Problem: {question}"})
    return tokenizer.apply_chat_template(messages, tokenize=False)


class StepParser:
    """Extract reasoning steps from raw LLM output – robust to messy formats."""
    # 1) "Step n:" / "Step One:" / "Step IV‑" …
    _STEP_RE = re.compile(
        r"^\s*Step\s*(?:\d+|[IVXLCDM]+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)\b\s*[:\-.]",
        re.I,)
    # 2) 1. / 1) / 1‑   OR  I. / II) / III‑
    _ENUM_RE = re.compile(r"^\s*(?:\d+|[IVXLCDM]+)[.)-]\s+", re.I)
    # 3) Narrative adverbs
    _NARRATIVE_RE = re.compile(
        r"^\s*(First|Firstly|Second|Secondly|Third|Thirdly|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|Next|Then|After that|Therefore|However|Since|Thus|Finally|Lastly|In conclusion)\b[,:]?",
        re.I,)
    # 4) Bullet mark (dash/asterisk/dot) + space
    _BULLET_RE = re.compile(r"^\s*[-*•]\s+")
    _JUNK_RE = re.compile(r"^(?:<\|endoftext\|>\s*)+")
    _DELIMITERS = (_STEP_RE, _ENUM_RE, _NARRATIVE_RE, _BULLET_RE)

    # ------------------------------------------------------------------
    @classmethod
    def _is_delimiter(cls, line: str) -> bool:
        return any(r.match(line) for r in cls._DELIMITERS)
    
    @classmethod
    def _clean_line(cls, line: str) -> str:
        """Strip junk tokens and surrounding whitespace from *line*."""
        return cls._JUNK_RE.sub("", line).strip()

    @classmethod
    def _split(cls, text: str) -> List[str]:
        """Split *text* into tentative step blocks using delimiter lines."""
        blocks: List[str] = []
        buf: List[str] = []

        def flush():
            if buf:
                blocks.append(" ".join(buf).strip())
                buf.clear()

        for raw_ln in text.splitlines():
            ln = cls._clean_line(raw_ln)
            if cls._is_delimiter(ln):
                flush()
            buf.append(ln)
        flush()
        return [b for b in blocks if b]
    
    @classmethod
    def _strip_leading_marker(cls, text: str) -> str:
        """Remove one leading delimiter + junk tokens from *text* if present."""
        txt = cls._clean_line(text)
        for p in cls._DELIMITERS:
            m = p.match(txt)
            if m:
                return txt[m.end():].lstrip()
        return txt

    @classmethod
    def parse(cls, text: str) -> List[str]:
        """Return **raw** steps (leading markers still present)."""
        steps = cls._split(text)
        if len(steps) >= 2:
            return steps
        # Fallback ①: blank‑line split
        sans_junk = "\n".join(cls._clean_line(l) for l in text.splitlines())
        paras = re.split(r"\n\s*\n", sans_junk)
        steps = [p.strip() for p in paras if p.strip()]
        if len(steps) >= 2:
            return steps
        # Fallback ②: line‑by‑line (already cleaned)
        return [ln for ln in sans_junk.splitlines() if ln.strip()]
    
    @classmethod
    def parse_clean(cls, obj: Union[str, Iterable[str]]) -> List[str]:
        """Return list of *clean* steps (leading markers removed)."""
        if isinstance(obj, str):
            raw_steps = cls.parse(obj)
        else:
            raw_steps = list(obj)
        return [cls._strip_leading_marker(s) for s in raw_steps]
