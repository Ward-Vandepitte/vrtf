"""
Shared LaTeX utilities for the OpenBooks OCR pipeline.

Provides text stripping and formula-aware parsing used by both
quality_evaluation_service and page_recreation_service.
"""

from __future__ import annotations

import re


def _extract_brace_arg(text: str, pos: int) -> tuple[str, int]:
    """Extract content of a brace-delimited argument starting at pos.

    Handles nested braces via depth counting.  Expects text[pos] == '{'.

    Returns (content_inside_braces, index_after_closing_brace).
    If no opening brace at pos, returns ('', pos).
    """
    if pos >= len(text) or text[pos] != '{':
        return '', pos
    depth = 0
    start = pos + 1
    j = pos
    while j < len(text):
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0:
                return text[start:j], j + 1
        j += 1
    # Unmatched brace — return what we have
    return text[start:], len(text)


# --- Greek letter mapping (Unicode Greek for bitmap/PIL rendering) ---
_GREEK_MAP: dict[str, str] = {
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
    'epsilon': 'ε', 'varepsilon': 'ε', 'zeta': 'ζ', 'eta': 'η',
    'theta': 'θ', 'vartheta': 'θ', 'iota': 'ι', 'kappa': 'κ',
    'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ',
    'pi': 'π', 'varpi': 'ϖ', 'rho': 'ρ', 'varrho': 'ρ',
    'sigma': 'σ', 'varsigma': 'ς', 'tau': 'τ', 'upsilon': 'υ',
    'phi': 'φ', 'varphi': 'φ', 'chi': 'χ', 'psi': 'ψ', 'omega': 'ω',
    'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ', 'Lambda': 'Λ',
    'Xi': 'Ξ', 'Pi': 'Π', 'Sigma': 'Σ', 'Upsilon': 'Υ',
    'Phi': 'Φ', 'Psi': 'Ψ', 'Omega': 'Ω',
}

# Commands that just unwrap their argument: \mathrm{X} → X
_UNWRAP_CMDS = frozenset({
    'mathrm', 'mathbf', 'mathit', 'mathsf', 'mathtt', 'mathcal',
    'mathfrak', 'mathbb', 'mathscr',
    'boldsymbol', 'operatorname', 'textbf', 'textit', 'textrm', 'text',
    'widehat', 'overline', 'underline', 'widetilde', 'hat', 'bar',
    'tilde', 'vec', 'dot', 'ddot', 'acute', 'grave', 'check', 'breve',
    'underbrace', 'overbrace', 'mbox', 'hbox',
    'displaystyle', 'textstyle', 'scriptstyle', 'scriptscriptstyle',
})

# Math functions rendered as plain words
_MATH_FUNCS = frozenset({
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
    'arcsin', 'arccos', 'arctan',
    'sinh', 'cosh', 'tanh', 'coth',
    'log', 'ln', 'exp', 'lim', 'max', 'min', 'sup', 'inf',
    'det', 'dim', 'ker', 'arg', 'deg', 'gcd', 'hom', 'mod',
})

# Symbol replacements: \cmd → string
_SYMBOL_MAP: dict[str, str] = {
    'times': 'x', 'cdot': '.', 'cdots': '...', 'ldots': '...',
    'leq': '<=', 'geq': '>=', 'neq': '!=', 'approx': '~',
    'equiv': '=', 'pm': '+-', 'mp': '-+',
    'infty': 'inf', 'partial': 'd',
    'nabla': 'V', 'forall': 'A', 'exists': 'E',
    'in': 'in', 'notin': 'notin', 'subset': 'c', 'supset': ')',
    'cup': 'U', 'cap': 'n',
    'rightarrow': '->', 'leftarrow': '<-', 'Rightarrow': '=>',
    'leftrightarrow': '<->', 'to': '->',
    'int': 'int', 'sum': 'sum', 'prod': 'prod',
    'oint': 'int', 'iint': 'int', 'iiint': 'int',
    'quad': ' ', 'qquad': '  ', 'enspace': ' ',
    ',': ' ', ';': ' ', '!': '', ':': ' ',
    'circ': 'o', 'lrcorner': '', 'bullet': '.', 'colon': ':',
}

# Delimiter commands: \left( → (, \right) → ), \big( → (, etc.
_DELIM_CMDS = frozenset({
    'left', 'right', 'big', 'Big', 'bigg', 'Bigg',
    'bigl', 'bigr', 'Bigl', 'Bigr', 'biggl', 'biggr', 'Biggl', 'Biggr',
})


def simplify_latex(text: str) -> str:
    """Convert LaTeX markup back to flat characters.

    Unlike strip_latex() which removes all formula content, this function
    preserves the character sequence that a 1960s typewriter would have
    printed — e.g. ``\\frac{a}{b}`` becomes ``a/b``.

    If the result is empty but the original had content, falls back to
    strip_latex() output.
    """
    if not text:
        return ''

    original_len = len(text.strip())

    # Step 1: Strip $ / $$ delimiters, keeping content (NO re.DOTALL)
    result = re.sub(r'\$\$(.*?)\$\$', r'\1', text)
    result = re.sub(r'\$(.*?)\$', r'\1', result)

    # Step 2: Process backslash commands in a single left-to-right pass
    out: list[str] = []
    i = 0
    n = len(result)

    while i < n:
        ch = result[i]

        if ch != '\\':
            # Not a command — pass through
            # But handle sub/superscript braces: _{X} → X, ^{X} → X
            if ch in ('_', '^') and i + 1 < n and result[i + 1] == '{':
                content, end = _extract_brace_arg(result, i + 1)
                out.append(simplify_latex(content))
                i = end
            elif ch in ('_', '^') and i + 1 < n:
                # Single char sub/super: _X → X
                out.append(result[i + 1])
                i += 2
            else:
                out.append(ch)
                i += 1
            continue

        # Backslash at position i
        # Capture command name
        j = i + 1
        if j >= n:
            i += 1
            continue

        # Non-alpha after backslash: \\ → newline, \{ \} etc.
        if not result[j].isalpha():
            if result[j] in ('{', '}'):
                out.append(result[j])
            elif result[j] == '\\':
                out.append('\n')
            elif result[j] == ',':
                out.append(' ')
            elif result[j] == ' ':
                out.append(' ')
            else:
                out.append(result[j])
            i = j + 1
            continue

        # Extract command name
        while j < n and result[j].isalpha():
            j += 1
        cmd = result[i + 1:j]

        # Skip optional whitespace before possible brace arg
        k = j
        while k < n and result[k] == ' ':
            k += 1

        # \frac{A}{B} → A/B (also \cfrac, \dfrac, \tfrac)
        if cmd in ('frac', 'cfrac', 'dfrac', 'tfrac'):
            if k < n and result[k] == '{':
                num, after_num = _extract_brace_arg(result, k)
                # Skip whitespace between args
                while after_num < n and result[after_num] == ' ':
                    after_num += 1
                if after_num < n and result[after_num] == '{':
                    den, after_den = _extract_brace_arg(result, after_num)
                    out.append(f'{simplify_latex(num)}/{simplify_latex(den)}')
                    i = after_den
                else:
                    out.append(simplify_latex(num))
                    i = after_num
            else:
                i = j
            continue

        # \sqrt{X} → sqrt(X)
        if cmd == 'sqrt':
            # Optional [n] for nth root
            sq_pos = k
            if sq_pos < n and result[sq_pos] == '[':
                bracket_end = result.find(']', sq_pos)
                if bracket_end != -1:
                    sq_pos = bracket_end + 1
                    while sq_pos < n and result[sq_pos] == ' ':
                        sq_pos += 1
            if sq_pos < n and result[sq_pos] == '{':
                content, after = _extract_brace_arg(result, sq_pos)
                out.append(f'sqrt({simplify_latex(content)})')
                i = after
            else:
                out.append('sqrt')
                i = j
            continue

        # \begin{...} → empty, consuming env name + optional [pos] + ONE {col-spec}
        if cmd == 'begin':
            if k < n and result[k] == '{':
                _, after = _extract_brace_arg(result, k)
                # Skip optional [...] positional arg
                while after < n and result[after] == ' ':
                    after += 1
                if after < n and result[after] == '[':
                    bracket_end = result.find(']', after)
                    if bracket_end != -1:
                        after = bracket_end + 1
                # Skip ONE optional {column-spec} arg
                while after < n and result[after] == ' ':
                    after += 1
                if after < n and result[after] == '{':
                    _, after = _extract_brace_arg(result, after)
                i = after
            else:
                i = j
            continue
        if cmd == 'end':
            if k < n and result[k] == '{':
                _, after = _extract_brace_arg(result, k)
                i = after
            else:
                i = j
            continue

        # Unwrap commands: \mathrm{X} → X, \widehat{X} → X, etc.
        if cmd in _UNWRAP_CMDS:
            if k < n and result[k] == '{':
                content, after = _extract_brace_arg(result, k)
                out.append(simplify_latex(content))
                i = after
            else:
                i = j
            continue

        # Delimiter commands: \left( → (, \right) → ), etc.
        if cmd in _DELIM_CMDS:
            if k < n and result[k] in ('(', ')', '[', ']', '{', '}', '|', '.', '\\'):
                if result[k] == '.':
                    pass  # \left. = invisible delimiter
                elif result[k] == '\\':
                    # \left\{ etc — skip the backslash, use next char
                    if k + 1 < n and result[k + 1] in ('{', '}', '|'):
                        out.append(result[k + 1])
                        i = k + 2
                        continue
                else:
                    out.append(result[k])
                i = k + 1
            else:
                i = j
            continue

        # Greek letters
        if cmd in _GREEK_MAP:
            out.append(_GREEK_MAP[cmd])
            i = j
            continue

        # Math functions: \sin → sin, etc.
        if cmd in _MATH_FUNCS:
            out.append(cmd)
            i = j
            continue

        # Symbol map
        if cmd in _SYMBOL_MAP:
            out.append(_SYMBOL_MAP[cmd])
            i = j
            continue

        # \overset{above}{base} → base, \underset{below}{base} → base
        if cmd in ('overset', 'underset', 'stackrel'):
            if k < n and result[k] == '{':
                _, after_first = _extract_brace_arg(result, k)
                while after_first < n and result[after_first] == ' ':
                    after_first += 1
                if after_first < n and result[after_first] == '{':
                    base, after_base = _extract_brace_arg(result, after_first)
                    out.append(simplify_latex(base))
                    i = after_base
                else:
                    i = after_first
            else:
                i = j
            continue

        # Catch-all: \cmd{X} → X (unknown command with brace arg)
        if k < n and result[k] == '{':
            content, after = _extract_brace_arg(result, k)
            out.append(simplify_latex(content))
            i = after
            continue

        # Bare \cmd without braces → empty
        i = j

    result = ''.join(out)

    # Clean up: remove stray {}, normalize whitespace per line
    result = result.replace('{', '').replace('}', '')
    lines = result.split('\n')
    lines = [re.sub(r'[^\S\n]+', ' ', line).strip() for line in lines]
    result = '\n'.join(line for line in lines if line)

    # Fallback: if result is empty but original had content
    if not result and original_len > 0:
        result = strip_latex(text)

    return result


def strip_latex(text: str) -> str:
    """Strip inline LaTeX commands from text, preserving readable content.

    Removes:
    - $$...$$ display math
    - $...$ inline math
    - \\cmd{...} command patterns
    - Remaining \\commands
    """
    # Remove $...$ and $$...$$ blocks
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$', '', text)
    # Remove \cmd{...} patterns
    text = re.sub(r'\\[a-zA-Z]+\s*\{[^}]*\}', '', text)
    # Remove remaining \commands
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
