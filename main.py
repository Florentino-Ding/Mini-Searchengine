import os

from base import SearchEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
os.chdir(BASE_DIR)
DEFAULT_CONTINUE = False

search_engine = SearchEngine(mode="debug")
search_engine.warm_up()


def shell_evaluate(query: str) -> list[str]:
    return search_engine.get_result_urls(query)


if __name__ == "__main__":
    while True:
        query = input(">>> ")
        if query == "exit":
            break
        urls = shell_evaluate(query)
        for url in urls:
            print(url)
