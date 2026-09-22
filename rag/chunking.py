"""Markdown hierarchy and balanced formula aware recursive splitting."""
import re
from bisect import bisect_left, bisect_right
from collections.abc import Callable

from langchain_core.documents import Document


def estimate_tokens(text: str) -> int:
    """Conservative fallback; production supplies the embedding tokenizer."""
    return len(re.findall(r'[\u4e00-\u9fff]|\w+|[^\w\s]', text))


class RuleChunker:
    def __init__(self, chunk_size: int = 500, count_tokens: Callable = estimate_tokens):
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive')
        self.chunk_size = chunk_size
        self.count_tokens = count_tokens

    @staticmethod
    def _boundaries(text):
        """Only split outside brackets, fenced code and formula lines."""
        stack, cuts = [], {0: 0, len(text): 0}
        pairs = {')': '(', '）': '（', ']': '[', '】': '【', '}': '{'}
        fence = None
        offset = 0
        for line in text.splitlines(keepends=True):
            marker = re.match(r'^\s*(`{3,}|~{3,})', line)
            if marker:
                token = marker.group(1)
                if fence is None:
                    fence = token
                elif token[0] == fence[0] and len(token) >= len(fence):
                    fence = None
                offset += len(line)
                if fence is None and not stack:
                    cuts[offset] = 0
                continue
            if fence is not None:
                offset += len(line)
                continue
            formula = bool(re.search(r'=|\\\[|\$\$', line))
            for i, char in enumerate(line):
                if char in pairs.values():
                    stack.append(char)
                elif char in pairs:
                    if not stack or stack.pop() != pairs[char]:
                        raise ValueError('Unbalanced parentheses/brackets in rule document')
                if not stack and not formula:
                    pos = offset + i + 1
                    sentence_end = char == '.' and (i + 1 == len(line) or line[i + 1].isspace())
                    if char in '。；;!?！？' or sentence_end:
                        cuts[pos] = 1
                    elif char in '，,、' or char.isspace():
                        cuts[pos] = 2
                    elif char in '且或':
                        cuts[pos] = 2
                    else:
                        cuts[pos] = 3
            offset += len(line)
            if not stack:
                cuts[offset] = 0 if not line.strip() else 1
        if stack or fence is not None:
            raise ValueError('Unclosed bracket or fenced code block in rule document')
        return cuts

    def _split(self, text):
        cuts = self._boundaries(text)
        levels = {level: sorted(p for p, value in cuts.items() if value == level) for level in range(4)}

        def descend(start, end, priority):
            if self.count_tokens(text[start:end]) <= self.chunk_size or priority > 3:
                return [text[start:end]]
            positions = levels[priority]
            positions = positions[bisect_right(positions, start):bisect_left(positions, end)]
            if not positions:
                return descend(start, end, priority + 1)
            pieces, left = [], start
            for right in [*positions, end]:
                pieces.extend(descend(left, right, priority + 1))
                left = right
            # Merge small siblings without crossing an unsafe formula boundary.
            merged, current = [], ''
            for piece in pieces:
                if current and self.count_tokens(current + piece) > self.chunk_size:
                    merged.append(current)
                    current = ''
                current += piece
            if current:
                merged.append(current)
            return merged

        return descend(0, len(text), 0)

    def split(self, text: str, *, parent_id: str, title: str = '', metadata=None):
        safe_boundaries = self._boundaries(text)
        offset = 0
        sections, hierarchy, buffer = [], [], []
        fence = None
        for line in text.splitlines(keepends=True):
            marker = re.match(r'^\s*(`{3,}|~{3,})', line)
            if marker:
                token = marker.group(1)
                if fence is None:
                    fence = token
                elif token[0] == fence[0] and len(token) >= len(fence):
                    fence = None
            heading = re.match(r'^(#{1,6})\s+(.+?)\s*#*\s*$', line) if fence is None and offset in safe_boundaries else None
            offset += len(line)
            if heading:
                if buffer:
                    sections.append((list(hierarchy), ''.join(buffer)))
                level = len(heading.group(1))
                hierarchy = [(n, h) for n, h in hierarchy if n < level]
                hierarchy.append((level, heading.group(2)))
                buffer = [line]
            else:
                buffer.append(line)
        if buffer:
            sections.append((list(hierarchy), ''.join(buffer)))
        documents = []
        for path, content in sections:
            for chunk in self._split(content):
                if not chunk.strip():
                    continue
                headings = [h for _, h in path]
                documents.append(Document(page_content=chunk, metadata={
                    **(metadata or {}), 'parent_id': parent_id,
                    'heading_path': ' > '.join(headings), 'heading_level': path[-1][0] if path else 0,
                    'summary': ' / '.join([title, *headings]).strip(' /'),
                    'formula_type': 'conditional' if re.search(r'如果|若|\bif\b|\belse\b', chunk, re.IGNORECASE) else ('arithmetic' if '=' in chunk else 'prose'),
                    'token_count': self.count_tokens(chunk),
                    'oversized': self.count_tokens(chunk) > self.chunk_size,
                }))
        return documents
