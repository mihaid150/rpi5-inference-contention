import re


class StableChunker:

    def __init__(self, token_count, tokenizer_id, max_tokens=256):
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("Invalid token budget")
        self.token_count = token_count
        self.max_tokens = max_tokens
        self.identity = dict(
            method="markdown-paragraph-heading-v1",
            max_tokens=max_tokens,
            tokenizer_id=tokenizer_id,
            oversized="recursive-whitespace-then-character",
            overlap=0,
        )

    def __call__(self, document):
        headings, lines, chunks = ([], [], [])
        fence = None

        def emit(body):
            body = body.strip()
            if not body:
                return
            prefix = "\n".join((text for _, text in headings))
            prefix = prefix + "\n\n" if prefix else ""
            if self.token_count(prefix) >= self.max_tokens:
                raise ValueError("Heading context exhausts token budget")

            def split(text):
                combined = prefix + text
                if self.token_count(combined) <= self.max_tokens:
                    chunks.append(combined)
                    return
                if len(text) <= 1:
                    raise ValueError("Cannot fit body under token budget")
                midpoint = len(text) // 2
                spaces = [
                    m.start()
                    for m in re.finditer("\\s+", text)
                    if 0 < m.start() < len(text) - 1
                ]
                cut = (
                    min(spaces, key=lambda x: abs(x - midpoint)) if spaces else midpoint
                )
                split(text[:cut])
                split(text[cut:])

            split(body)

        def flush():
            emit("\n".join(lines))
            lines.clear()

        for line in document.replace("\r\n", "\n").split("\n"):
            marker = re.match("^\\s{0,3}(`{3,}|~{3,})", line)
            if marker:
                if fence is None:
                    flush()
                    fence = marker.group(1)
                elif marker.group(1)[0] == fence[0] and len(marker.group(1)) >= len(
                    fence
                ):
                    lines.append(line)
                    flush()
                    fence = None
                    continue
            heading = (
                re.match("^\\s{0,3}(#{1,6})\\s+(.+)$", line) if fence is None else None
            )
            if heading:
                flush()
                level = len(heading.group(1))
                headings[:] = [(n, t) for n, t in headings if n < level]
                headings.append((level, line.strip()))
                if self.token_count(line.strip()) > self.max_tokens:
                    raise ValueError("Heading exceeds token budget")
                chunks.append(line.strip())
            elif not line.strip() and fence is None:
                flush()
            else:
                lines.append(line)
        flush()
        return chunks
