import lxml.html
import structlog
import tiktoken
from .models import get_model

logger = structlog.get_logger("scrapeghost")


def _tostr(obj: lxml.html.HtmlElement) -> str:
    """
    Given lxml.html.HtmlElement, return string
    """
    return lxml.html.tostring(obj, encoding="unicode")


def _tokens(model: str, html: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(html))


def cost_estimate(html: str, model: str = "gpt-4") -> float:
    """
    Given HTML, return cost estimate in dollars.

    This is a very rough estimate and not guaranteed to be accurate.
    """
    tokens = _tokens(model, html)
    model_data = get_model[model]
    # assumes response is half as long as prompt, which is probably wrong
    return model_data.cost(tokens, tokens // 2)
