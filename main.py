import os

from base import SearchEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
os.chdir(BASE_DIR)
DEFAULT_CONTINUE = True

engine = None


def warm_up_for_shell() -> None:
    global engine
    engine = SearchEngine(
        [""],
        [""],
        "html",
        "cache",
        "stopwords/baidu_stopwords.txt",
        "search",
    )
    engine.warm_up_for_shell()


def warm_up_for_web() -> None:
    global engine
    engine = SearchEngine(
        [""],
        [""],
        "html",
        "cache",
        "stopwords/baidu_stopwords.txt",
        "debug",
    )
    engine.warm_up_for_web()


def my_evaluate(query: str) -> list[str]:
    global engine
    assert engine is not None
    return engine.search(query)


def ui_evaluate(query: str) -> list[dict[str, str]]:
    global engine
    assert engine is not None
    if not query:
        return []
    return engine.search_for_web(query)


if __name__ == "__main__":
    warm_up_for_web()
    print(ui_evaluate("人工智能"))
