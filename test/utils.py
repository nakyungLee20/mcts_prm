import sympy as sp
import math
from typing import Optional
import re

def _extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content from \\boxed{...} with proper brace balancing"""
    import re
    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    start_match = matches[-1]  # Use last occurrence (final answer)
    start_pos = start_match.end() - 1  # Position of opening brace
    brace_count = 0
    pos = start_pos
    while pos < len(text):
        if text[pos] == '{':
            brace_count += 1
        elif text[pos] == '}':
            brace_count -= 1
            if brace_count == 0:
                content = text[start_pos + 1:pos]
                return content.strip()
        pos += 1
    return None 

def _strip_markup_enhanced(ans: str) -> str:
    """Enhanced markup removal with better LaTeX handling"""
    # Remove LaTeX display math wrappers \[ … \] or $$ … $$
    ans = re.sub(r"\\\[.*?\\\]", "", ans)
    ans = re.sub(r"\$\$.*?\$\$", "", ans)
    # Remove inline LaTeX: \( ... \) and $...$
    ans = re.sub(r"\\\((.*?)\\\)", r"\1", ans)
    ans = re.sub(r"\$(.*?)\$", r"\1", ans)
    # Handle \boxed{...} with proper brace balancing
    boxed_content = _extract_boxed_answer(f"\\boxed{{{ans}}}")
    if boxed_content:
        ans = boxed_content
    # Remove common LaTeX commands but preserve fractions
    ans = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", ans)  # \text{...}
    ans = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", ans)  # \mathrm{...}
    # Convert LaTeX fractions to evaluable form: \frac{a}{b} -> (a)/(b)
    ans = re.sub(r"\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}", r"(\1)/(\2)", ans)
    # Remove remaining LaTeX commands
    ans = re.sub(r"\\[a-zA-Z]+\*?", "", ans)
    # Remove variable assignments like "k =" or "x=" at start
    ans = re.sub(r"^[a-zA-Z]\s*=\s*", "", ans)
    # Clean up extra whitespace and punctuation
    ans = ans.strip()
    if ans.startswith("$") and ans.endswith("$"):
        ans = ans[1:-1]
    
    return ans.strip()

def _sanitize_enhanced(text: str) -> str:
    """Enhanced normalization with better numeric handling"""
    text = _strip_markup_enhanced(text)
    text = text.strip()
    # Remove trailing punctuation
    text = re.sub(r"[\s\.;:,]+$", "", text)
    # Normalize spaces
    text = re.sub(r"\s+", " ", text)
    # Handle negative signs and spaces
    text = re.sub(r"\s*-\s*", "-", text)
    return text

def _to_float_enhanced(expr: str) -> Optional[float]:
    """Enhanced numeric evaluation with fraction support"""
    try:
        # Handle simple cases first
        if expr.replace(".", "").replace("-", "").isdigit():
            return float(expr)
        
        # Handle fractions: -33/2, 33/2, etc.
        if re.match(r"^-?\d+/\d+$", expr):
            parts = expr.split("/")
            return float(parts[0]) / float(parts[1])
        
        # Handle parenthetical fractions: (-33)/(2)
        paren_match = re.match(r"^\(([^)]+)\)/\(([^)]+)\)$", expr)
        if paren_match:
            num, den = paren_match.groups()
            return float(num) / float(den)
        
        # Try eval for more complex expressions
        safe_expr = expr.replace("^", "**")
        return float(eval(safe_expr))
        
    except Exception:
        return None

def _numeric_equiv_enhanced(a: str, b: str) -> bool:
    """Enhanced numeric equivalence with better fraction handling"""
    a_clean, b_clean = map(_sanitize_enhanced, (a, b))
    # Exact string match first
    if a_clean == b_clean:
        return True

    # Numeric comparison
    a_val, b_val = _to_float_enhanced(a_clean), _to_float_enhanced(b_clean)
    if a_val is not None and b_val is not None:
        return math.isclose(a_val, b_val, rel_tol=1e-6, abs_tol=1e-9)

    # SymPy fallback for symbolic expressions
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
        prompt = """You are a math expert. Solve the problem step-by-step. Use format: "Step k: [reasoning]" then "Answer: [result]". STOP after Answer.

**Rules:**
- Write "Step k: " before each step
- Keep steps concise and focused
- Write "Answer: [result]" and STOP immediately
- No explanations after Answer
- Maximum 5-7 steps for most problems

Format Guide with Examples:
**Example:**
Problem: Find the sum of the first 8 positive even integers.
Step 1: The first 8 even integers are 2, 4, 6, 8, 10, 12, 14, 16.
Step 2: Use the formula for an arithmetic series: S = n·(first + last)/2.
Step 3: Substitute n=8, first=2, last=16 to get S = 8·(2+16)/2 = 8·9 = 72.
Answer: 72

**Example:**
Problem: Next number in 2,4,8,16?
Step 1: Pattern: multiply by 2 each time
Step 2: 16 × 2 = 32
Answer: 32

Remember: Write Answer and STOP. No extra text."""
    
    if type == "rollout":
        prompt = """You are a math expert. Continue solving from the given steps. Use format: "Step k+1: [reasoning]" then "Answer: [result]". STOP after Answer.

**Rules:**
- Continue with "Step k+1: ", "Step k+2: ", etc.
- Keep steps concise and focused  
- Write "Answer: [result]" and STOP immediately
- No explanations after Answer
- Complete solution efficiently

**Example:**
Current steps:
Problem: Find sum of first 8 even integers.
Step 1: First 8 even integers: 2,4,6,8,10,12,14,16
Step 2: Use formula S = n(first+last)/2
Continue and finish:
Step 3: S = 8(2+16)/2 = 8×9 = 72
Answer: 72

**Example:**
Current steps:
Problem: Next number in 2,4,8,16?
Step 1: Pattern: multiply by 2 each time
Continue and finish:
Step 2: 16 × 2 = 32
Answer: 32

Remember: Write Answer and STOP. No extra text."""
    return prompt
