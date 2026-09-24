"""Conservative lexical integrity guards; semantic interpretation still needs evidence.

These labels describe observed syntax, not a formally parsed policy or verified eligibility.
"""
import re

PATTERNS = {
    'eligibility': r'\b(if|only|provided|requires?|eligible|qualif\w*|at least|at most)\b|如果|若|仅限|满足|条件',
    'exception': r'\b(except|unless|excluding|excluded|ineligible|however)\b|除外|除非|不适用|但不|例外',
    'negation': r'\b(not|never|cannot|no discount|nonmembers?)\b|不得|不享受|不可|非会员',
    'conjunction': r'\b(and|both)\b|并且|且|同时',
    'alternative': r'\b(or|either|whichever|greater|better)\b|或者|或|二选一',
    'illustration': r'\b(example|illustration|illustrative|e\.g\.)|例如|举例|示例',
    'arithmetic': r'=|\b(percent|percentage|subtract|multiply|round|cap|floor)\b|计算|公式|封顶|取整',
    'parenthetical': r'[()（）\[\]【】{}｛｝［］「」『』]',
}


def condition_kinds(text):
    return [kind for kind, pattern in PATTERNS.items() if re.search(pattern, text, re.IGNORECASE)]


def protected_logic_spans(text):
    """Keep a condition paragraph/list with its main clause and attached exception.

Markdown headings/blank lines delimit paragraphs. An exception/continuation after a
blank line attaches to the preceding paragraph; ordinary prose can still be split.
Overlong protected blocks are flagged by RuleChunker rather than cut into half-rules.
"""
    blocks, start, end, offset = [], None, None, 0
    for line in text.splitlines(keepends=True):
        if not line.strip() or re.match(r'^#{1,6}\s', line):
            if start is not None:
                blocks.append((start, end))
                start = end = None
        else:
            if start is None:
                start = offset
            end = offset + len(line)
        offset += len(line)
    if start is not None:
        blocks.append((start, end))
    spans = []
    for i, (start, end) in enumerate(blocks):
        block = text[start:end]
        if condition_kinds(block):
            # Do not detach a qualification on a separate paragraph from its antecedent.
            continuation = re.match(r'^\s*(?:[-*]\s*)?(?:except\b|unless\b|however\b|only if\b|but\b|note\b|除外|除非|但是|注意)', block, re.IGNORECASE)
            if continuation and i and not re.search(r'^#{1,6}\s', text[blocks[i-1][1]:start], re.MULTILINE):
                start = blocks[i-1][0]
            if spans and start <= spans[-1][1]:
                spans[-1] = (spans[-1][0], end)
            else:
                spans.append((start, end))
    return spans


SCOPE_INSTRUCTIONS = '''
Interpret policy syntax with scope: arithmetic parentheses group operations; natural-language
parentheses may restrict the immediately preceding benefit, state an exception, or give an example.
Do not treat every parenthetical amount as an extra discount. Examples are illustrative unless an
operative rule makes them binding. Preserve nested AND/OR grouping and negation: A AND (B OR C)
is different from (A AND B) OR C. "Unless", "except", "only if", "not", caps, floors and per-order limits
can override a general benefit. Carry attached notes and inherited section scope with the main rule.
Do not repair unmatched brackets by guessing. If scope, precedence or stacking is ambiguous, ask for
clarification and do not return a computed price. Do not invent an arbitrary precedence for conflicting policies.
'''
