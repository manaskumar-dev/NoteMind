import markdown
import bleach

ALLOWED_TAGS = [
    "p", "br", "strong", "em", "ul", "ol", "li", "blockquote", "code", "pre",
    "h1", "h2", "h3", "h4", "a", "hr", "table", "thead", "tbody", "tr", "th", "td",
]
ALLOWED_ATTRS = {"a": ["href", "title", "rel"]}


def render_markdown_safe(raw_md: str) -> str:
    """Converts markdown to HTML, then strips any tag/attr not on the allowlist.
    This is the only place user content becomes HTML, preventing stored XSS.
    """
    html = markdown.markdown(raw_md, extensions=["fenced_code", "tables"])
    return bleach.clean(html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)
