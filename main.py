import os

from utils import get_args
from index_processor import WebCrawl, TextTransformer, IndexMaker


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
os.chdir(BASE_DIR)


if __name__ == "__main__":
    arg_dict = get_args()
    t = TextTransformer(arg_dict["html_dir"], "cache", "cache", "cache", "stopwords")
    index_maker = IndexMaker("cache")
    if not t.load_html(arg_dict["html_dir"], cache_dir="cache"):
        print("No html files found.", end=" ")
        input("Press any key to start crawling...")
        for start_url in arg_dict["page"]:
            robot = WebCrawl(
                start_url,
                arg_dict["scope"],
                HEADERS,
            )
            robot.run()
    assert t.load_html_from_memory()
    t.make_term_index("cache")
    t.load_stopwords("stopwords/baidu_stopwords.txt")
