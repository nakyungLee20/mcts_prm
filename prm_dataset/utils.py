import sympy as sp
from typing import Optional
import re
import math

def _strip_markup(ans: str) -> str:
    """Remove common LaTeX/markup & variable tags."""
    # # Remove LaTeX inline math wrappers \( … \) or \[ … \]
    ans = re.sub(r"\\\[.*?\\\]", "", ans)
    ans = re.sub(r"\$\$.*?\$\$", "", ans)
    # Remove inline LaTeX: \( ... \) and $...$
    ans = re.sub(r"\\\((.*?)\\\)", r"\1", ans)
    ans = re.sub(r"\$(.*?)\$", r"\1", ans)
    # Remove \boxed{...}
    ans = re.sub(r"\\boxed\s*{([^}]*)}", r"\1", ans)
    # Remove LaTeX commands like \text{...}, \frac{...}, etc.
    ans = re.sub(r"\\[a-zA-Z]+\s*(\{[^{}]*\})?", "", ans)
    # Remove variable assignments like "y =" or "x=" at start
    ans = re.sub(r"^[a-zA-Z]\s*=\s*", "", ans)
    # Trim outer $ … $ if present
    ans = ans.strip()
    if ans.startswith("$") and ans.endswith("$"):
        ans = ans[1:-1]
    return ans.strip()

def _sanitize(text: str) -> str:
    """Normalise a candidate answer string for comparison."""
    text = _strip_markup(text)
    text = text.strip()
    text = re.sub(r"[\s\.;:,]+$", "", text)     # trailing punctuation
    text = re.sub(r"\s+", " ", text)              # collapse spaces
    return text

def _to_float(expr: str) -> Optional[float]:
    try:
        return float(eval(expr.replace("^", "**")))
    except Exception:
        return None

def _numeric_equiv(a: str, b: str) -> bool:
    """Return True if `a` and `b` are numerically equivalent or exact match."""
    a_clean, b_clean = map(_sanitize, (a, b))
    if a_clean == b_clean:
        return True

    # Attempt simple numeric evaluation
    a_val, b_val = _to_float(a_clean), _to_float(b_clean)
    if a_val is not None and b_val is not None:
        return math.isclose(a_val, b_val, rel_tol=1e-6)

    if sp is not None:
        try:
            a_expr = sp.sympify(a_clean.replace("^", "**"))
            b_expr = sp.sympify(b_clean.replace("^", "**"))
            return sp.simplify(a_expr - b_expr) == 0
        except Exception:
            pass
    return False

def system_prompt(type):
    prompt = ""
    if type == "sample":
        prompt = """You are a math-problem expert. Your task is to complete the step-by-step solution for the problem provided. Write each reasoning step on its own line in the exact form \"Step k: [your reasoning step]\n\", numbering start from Step 1. When the final answer is obtained, write exactly one final line, \"Answer: [Final answer]\". Do NOT add explanations, extra steps, or any text after the "Answer:" line.

**Format Guide**: (You MUST write "Step " before numbering the step.)
Step 1: [Step 1 reasoning]\n
Step 2: [Step 2 reasoning]\n
...
Step k: [Step k reasoning]\n
...
Answer: [Final answer]

Format Guide with Examples:
<Example 1>
Problem: Find the sum of the first 8 positive even integers.
Step 1: The first 8 even integers are 2, 4, 6, 8, 10, 12, 14, 16.
Step 2: Use the formula for an arithmetic series: S = n·(first + last)/2.
Step 3: Substitute n=8, first=2, last=16 to get S = 8·(2+16)/2 = 8·9 = 72.
Answer: 72

<Example 2>
Problem: Determine the next number in the sequence 2, 4, 8, 16.
Step 1: Notice each term is obtained by multiplying the previous term by 2.
Step 2: Multiply 16 by 2, 16 * 2 = 32.
Answer: 32

Follow the FORMAT GUIDE structure exactly. Generate rationales step-by-step, not directly to the final answer. **Do NOT** write anything after the final 'Answer:' line. Always start stepwise reasoning with "Step {i-th}: " form."""
    if type == "rollout":
        prompt = """You are a math problem-solving expert. Continue solving the given problem step by step, strictly following the required format. Each new step must begin with \"Step k+1: ...\", \"Step k+2:...\", and so on, continuing from the last given step number. When the final answer is reached, write only one final line starting with: \"Answer: [Final Answer]\". Do not add any explanations, extra commentary, or additional text after the "Answer:" line. Your output must follow this exact step-by-step format with no deviations.

**Format Guide**: (You MUST write "Step " before numbering the step.)
Step 1: [Step 1 reasoning]\n
Step 2: [Step 2 reasoning]\n
...
Step k: [Step k reasoning]\n
Continue and finish the solution:
Step k+1: [Step k+1 reasoning]\n
...
Answer: [Final answer]

Format Guide with Examples:
<Example 1>
Current solution steps:
Problem: Find the sum of the first 8 positive even integers.
Step 1: The first 8 even integers are 2, 4, 6, 8, 10, 12, 14, 16.
Step 2: Use the formula for an arithmetic series: S = n·(first + last)/2.
Continue and finish the solution:
Step 3: Substitute n=8, first=2, last=16 to get S = 8·(2+16)/2 = 8·9 = 72.
Answer: 72

<Example 2>
Current solution steps:
Problem: Determine the next number in the sequence 2, 4, 8, 16.
Step 1: Notice each term is obtained by multiplying the previous term by 2.
Continue and finish the solution:
Step 2: Multiply 16 by 2, 16 * 2 = 32.
Answer: 32

Keep the reasoning steps precise and factual and complete the solution. Follow the FORMAT GUIDE structure exactly. **Do NOT** write anything after the final 'Answer:' line. Always start stepwise reasoning with "Step {i-th}: " form."""
    return prompt
