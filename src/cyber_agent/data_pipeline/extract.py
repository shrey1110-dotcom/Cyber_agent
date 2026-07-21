"""UTF-8 and HTML text extraction without network or browser dependencies."""

from __future__ import annotations

import re
from html.parser import HTMLParser


IGNORED_HTML_ELEMENTS = frozenset({"script", "style", "nav", "header", "footer", "aside", "menu", "form", "noscript"})
BLOCK_ELEMENTS = frozenset(
    {"article", "blockquote", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "main", "p", "pre", "section", "tr"}
)


class _ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        normalized = tag.lower()
        if normalized in IGNORED_HTML_ELEMENTS:
            self._ignored_depth += 1
        elif self._ignored_depth == 0 and normalized in BLOCK_ELEMENTS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in IGNORED_HTML_ELEMENTS and self._ignored_depth:
            self._ignored_depth -= 1
        elif self._ignored_depth == 0 and normalized in BLOCK_ELEMENTS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def validate_utf8(data: bytes) -> str:
    try:
        return data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"content is not valid UTF-8 at byte {exc.start}") from exc


def html_to_text(value: str) -> str:
    parser = _ReadableHTMLParser()
    try:
        parser.feed(value)
        parser.close()
    except Exception as exc:
        raise ValueError(f"malformed HTML: {exc}") from exc
    return parser.text()


def extract_text(raw_text: str, media_type: str) -> tuple[str, dict[str, float | str]]:
    if media_type in {"text/html", "application/xhtml+xml"}:
        markup_characters = sum(len(match.group(0)) for match in re.finditer(r"<[^>]*>", raw_text))
        ratio = markup_characters / max(1, len(raw_text))
        return html_to_text(raw_text), {"extraction": "html", "original_markup_ratio": round(ratio, 6)}
    if media_type.startswith("text/") or media_type in {"application/json", "application/yaml"}:
        return raw_text, {"extraction": "plain_text", "original_markup_ratio": 0.0}
    raise ValueError(f"unsupported media type: {media_type}")

