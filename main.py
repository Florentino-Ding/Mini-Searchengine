import os

from base import SearchEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
os.chdir(BASE_DIR)
DEFAULT_CONTINUE = True

engine = SearchEngine(
    [""],
    [""],
    "html",
    "cache",
    "stopwords/baidu_stopwords.txt",
)
engine.warm_up_for_search()


def my_evaluate(query: str) -> list[str]:
    return engine.search(query)


if __name__ == "__main__":
    while True:
        query = input("请输入查询语句：")
        if not query:
            print("无效的查询语句")
            continue
        result = my_evaluate(query)
        print(result)
