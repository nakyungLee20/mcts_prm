from datasets import load_dataset
import re, ast
from fractions import Fraction
from decimal import Decimal, InvalidOperation
from typing import Optional, Sequence, Tuple, List

class AnswerExtractor:
    """
    Robust gold-answer extractor for MATH-style datasets.
    """
    def __init__(self, *):
        self.answer_patterns = [
            r'(Answer:\s*[^\n]+)',
            r'(answer:\s*[^\n]+)',
            r'(Final Answer:\s*[^\n]+)',
            r'(final answer:\s*[^\n]+)',
            r'(Therefore:\s*[^\n]+)',
            r'(therefore:\s*[^\n]+)',
            r'(Result:\s*[^\n]+)',
            r'(result:\s*[^\n]+)',
            r'(The final answer is\s*[^\n]+)'
        ]
        self.number_patterns = [
            r'(\d+\.\d+)',   # 소수 (항상 정수보다 먼저)
            r'(\d+\/\d+)',   # 분수
            r'(\d+)',        # 정수
            r'(\d+\+\d+)',   # 덧셈
            r'(\d+\-\d+)',   # 뺄셈
            r'(\d+\*\d+)'    # 곱셈
        ]
        self.boxed_patterns = [
            r'\\boxed\{([^}]*)\}',
            r'\\fbox\{([^}]*)\}',
            r'\\boxed\s*\{([^}]*)\}',
            r'\\fbox\s*\{([^}]*)\}'
        ]

    def extract_pred_answer(self, text: str) -> Optional[str]:
        cleaned_text = self.remove_text_after_answer(text)
        # 1. 마지막 boxed 패턴 찾기
        boxed_answer = self.extract_last_boxed(cleaned_text)
        if boxed_answer:
            return self._strip_latex_delimiters(boxed_answer)
        # 2. Answer: 패턴 찾기
        answer_pattern = self.extract_answer_pattern(cleaned_text)
        if answer_pattern:
            return self._strip_latex_delimiters(answer_pattern)
        # 3. Latex 수식 번역하기
        for line in reversed(cleaned_text.splitlines()):
            line = line.strip()
            if not line:
                continue
            # 3-A) inline math delimiters $…$  \(…\)  \[…\]
            m = re.findall(r'\$(.*?)\$|\\\((.*?)\\\)|\\\[(.*?)\\\]', line)
            if m:
                # findall 은 튜플들의 리스트; 마지막 튜플의 첫 non-empty 요소
                expr = [seg for seg in m[-1] if seg][0]
                expr = self._strip_latex_delimiters(expr)
                if expr:
                    return expr
            # 3-B) bare \frac{…}{…} 나 \sqrt{…} 같은 명령어
            m2 = re.search(r'(\\[a-zA-Z]+(?:\{[^{}]+\})+)', line)
            if m2:
                return self._strip_latex_delimiters(m2.group(1))

            break    # 첫 non-empty line만 검사
        # 4. 마지막 숫자/수식 찾기
        final_number = self.extract_final_number(cleaned_text)
        if final_number:
            return self._strip_latex_delimiters(final_number)
        return None

    def extract_gold_answer(self, text: str, dataset: str | None = None) -> Optional[str]:
        # ───────── 0) Trivial ─────────
        if text is None:
            return None
        if isinstance(text, (list, tuple)):
            return self._flatten_and_clean(list(text))

        ds = (dataset or "").lower()
        txt = str(text).strip()

        # ───────── 1) GSM8K ─────────
        if ds == "gsm8k" or re.search(r"\n####\s*[^\n]+", txt):
            m = re.search(r"\n####\s*([^\n]+)", txt)
            return self._strip_latex_delimiters(m.group(1)) if m else None
        # ───────── 2) Math ─────────
        if ds == "math":
            return self.extract_pred_answer(txt)
        # ───────── 3) Omni (No need to extract) ─────────
        if ds == "omni":
            return self._strip_latex_delimiters(txt)
        # ───────── 4) OlympiadBench (list) ─────────
        if ds == "olympiad" or (txt.startswith('[') and txt.endswith(']')):
            try:
                parsed = ast.literal_eval(txt)
                if isinstance(parsed, (list, tuple)):
                    return self._flatten_and_clean(parsed)
                return self._strip_latex_delimiters(str(parsed))
            except (SyntaxError, ValueError):
                return self._strip_latex_delimiters(txt)
        # Fallback
        return self.extract_pred_answer(text)
    
    # Utils ─────────────────────────────────────────────────────
    def remove_text_after_answer(self, text: str) -> str:
        for pattern in self.answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                answer_part = match.group(1) # Answer: 부분까지만 유지하고 나머지 제거
                text = text[:text.find(answer_part) + len(answer_part)] # Answer: 이후의 모든 텍스트 제거
                break
        return text

    def extract_final_number(self, text: str) -> Optional[str]:
        lines = text.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            for pattern in self.number_patterns:
                matches = re.findall(pattern, line)
                if matches:
                    return matches[-1]  # 마지막 매치 반환
        return None

    def extract_last_boxed(self, text: str) -> Optional[str]:
        # 1) 마지막 \boxed{ 또는 \fbox{ 의 시작 위치 찾기
        start_pat = re.compile(r'(\\boxed|\\fbox)\s*\{')
        starts = list(start_pat.finditer(text))
        if not starts:
            return None
        # 2) 마지막 시작점부터 균형 잡힌 '}' 위치까지 스캔
        start_idx = starts[-1].end()       # { 바로 뒤 index
        i, depth = start_idx, 1
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:                     # 정상 종료
            content = text[start_idx : i-1]    # '}' 직전까지
            return content.strip()
        return None

    def extract_answer_pattern(self, text: str) -> Optional[str]:
        for pattern in self.answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                ans = re.sub(r'^[A-Za-z ]*[:\-]\s*', '', match.group(1))
                return ans.strip()
        return None

    def _strip_latex_delimiters(self, s: str | None) -> str | None:
        if s is None:
            return None
        s = s.strip()
        s = re.sub(r'^\$+\s*', '', s)           # 앞쪽 $
        s = re.sub(r'\s*\$+$', '', s)           # 뒤쪽 $
        s = re.sub(r'^\\\(|\\\)$', '', s)       # \( … \)
        s = re.sub(r'^\\\[|\\\]$', '', s)       # \[ … \]
        s = s.replace('\\\\', '\\')
        return s.strip(" ,;:")

    def _flatten_and_clean(self, items: List[str]) -> str:
        return ", ".join(self._strip_latex_delimiters(str(x)) for x in items)

