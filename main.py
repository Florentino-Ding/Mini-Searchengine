import os
import threading

from utils import get_args
from index_processing import crawl, text_transformer, indexing_processor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
os.chdir(BASE_DIR)


if __name__ == "__main__":
    arg_dict = get_args()
    t = text_transformer(arg_dict["html_dir"])
    if not t.load_html(arg_dict["html_dir"], cache_dir="cache/HTML.pkl"):
        print("No html files found.", end=" ")
        input("Press any key to start crawl.")
        for start_url in arg_dict["page"]:
            robot = crawl(start_url, HEADERS, arg_dict["depth"], arg_dict["thread"])
            robot.run()
        assert t.load_from_memory()
    t.make_dict("cache/word_dict.pkl")
