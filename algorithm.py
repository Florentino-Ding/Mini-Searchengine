import os
import threading
from typing import Literal, Optional
import bs4


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


def rank_page(
    content_list: dict[str, tuple[str, str]],
    method: Literal["keyword", "overlap", "meaning"] = "keyword",
) -> list[tuple[str, str, bs4.BeautifulSoup]]:
    raise NotImplementedError
