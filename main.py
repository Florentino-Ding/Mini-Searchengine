import os

from utils import get_args
from base import SearchEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
os.chdir(BASE_DIR)
DEFAULT_CONTINUE = True


def evaluate(query: str) -> list[str]:
    DEFAULT_CONTINUE = True
    engine = SearchEngine(
        [""],
        [""],
        "html",
        "cache",
        "stopwords/baidu_stopwords.txt",
    )
    engine._make_index()
    return engine.search(query, using_stopwords=False)


if __name__ == "__main__":
    pass
