import markdown
import bleach

ALLOWED_TAGS = [
    "p", "br", "strong", "em", "u", "mark", "ul", "ol", "li", "blockquote", "code", "pre",
    "h1", "h2", "h3", "h4", "a", "hr", "table", "thead", "tbody", "tr", "th", "td",
]
ALLOWED_HIGHLIGHT_CLASSES = {"hl-yellow", "hl-green", "hl-pink", "hl-blue"}


def _attribute_filter(tag, name, value):
    """Explicit allowlist per tag — never a blanket allow, never arbitrary style/class values."""
    if tag == "a":
        return name in ("href", "title", "rel")
    if tag == "mark":
        return name == "class" and value in ALLOWED_HIGHLIGHT_CLASSES
    return False


def render_markdown_safe(raw_md: str) -> str:
    """Converts markdown to HTML, then strips any tag/attr not on the allowlist.
    This is the only place user content becomes HTML, preventing stored XSS.
    """
    html = markdown.markdown(raw_md, extensions=["fenced_code", "tables"])
    return bleach.clean(html, tags=ALLOWED_TAGS, attributes=_attribute_filter, strip=True)