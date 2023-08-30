import os
import pickle
from typing import Optional, Union
import bs4
from url_normalize import url_normalize


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


def get_args() -> dict[str, Union[str, list[str], int]]:
    import sys
    import getopt

    try:
        opt_list, args = getopt.getopt(
            sys.argv[1:],
            "",
            ["target=", "page=", "scope=", "thread=", "html_dir="],
        )
    except getopt.GetoptError:
        print("GetoptError")
        sys.exit(1)
    arg_dict = dict()
    for opt, arg in opt_list:
        if opt == "--target":
            arg_dict["target"] = arg
        elif opt == "--page":
            arg = arg.split(",")
            arg_dict["page"] = arg
        elif opt == "--scope":
            arg = arg.split(",")
            arg_dict["scope"] = arg
        elif opt == "--html_dir":
            arg_dict["html_dir"] = arg
    assert "target" in arg_dict and "page" in arg_dict
    if "thread" not in arg_dict:
        arg_dict["thread"] = 0
    return arg_dict


def url_in_scope(url: str, scope: list[str]) -> bool:
    for item in scope:
        if item in url:
            return True
    return False


def url_norm(url: str) -> str:
    return url_normalize(url)  # type: ignore


def clean_front(url: str) -> str:
    if "http" in url:
        while not url.startswith("http"):
            url = url[1:]
        assert url.startswith("http://") or url.startswith("https://")
        return url
    else:
        if url.startswith("html/"):
            url = url[5:]
        elif url.startswith("/"):
            url = url[1:]
        return "http://" + url


def clean_rear(url: str) -> str:
    import re

    if url.endswith("html") or url.endswith("htm"):
        url_list = url.split("/")
        return "/".join(url_list[:-1]) + "/"
    elif url.endswith(".com") or url.endswith(".cn"):
        return url + "/"
    else:
        return url


def clean_file_format(url: str) -> str:
    return ".".join(url.split(".")[:-1])


def url_join(base_url: str, rear: Optional[str] = None) -> str:
    base_url = clean_rear(base_url)
    base_url = url_norm(base_url)
    if rear:
        # rear = clean_rear(rear)
        if rear.startswith("/"):
            rear = rear[1:]
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        return base_url + "/" + rear
    else:
        return base_url


def simplifyPath(path: str) -> str:
    assert "http" in path
    stack = []
    rear = "/" if path.endswith("/") else ""
    while not path.startswith("http"):
        path = path[1:]
    if path.startswith("https://"):
        path = path[8:]
        protocol = "https://"
    elif path.startswith("http://"):
        path = path[7:]
        protocol = "http://"
    else:
        raise ValueError("Invalid protocol")
    path_stack = path.split("/")

    for item in path_stack:
        if item == "..":
            if stack:
                stack.pop()
        elif item and item != ".":
            stack.append(item)
    return protocol + "/".join(stack) + rear


def get_title(html: bs4.BeautifulSoup):
    if html.head is None:
        return ""
    return html.head.text


def get_text(html: bs4.BeautifulSoup) -> str:
    return html.text


def process_text(text: str) -> str:
    text = text.replace("\n", "")
    text = text.replace("\t", "")
    text = text.replace(" ", "")
    return text


def process_html(html: bs4.BeautifulSoup) -> tuple[str, str]:
    head = get_title(html)
    text = get_text(html)
    return process_text(head), process_text(text)


def save_content(page_list: dict[str, bs4.BeautifulSoup], dir: str = "") -> None:
    page_content = dict[str, tuple[str, str]]()
    for url, html in page_list.items():
        title, text = process_html(html)
        page_content[url] = (title, text)
    pickle.dump(page_content, open(dir + "content.pkl", "wb"))


def read_content(dir: str = "") -> dict[str, tuple[str, str]]:
    assert os.path.exists(dir + "/content.pkl")
    return pickle.load(open(dir + "/content.pkl", "rb"))
