import os
from typing import Optional, Union
import requests
import bs4


HTML = dict[str, bs4.BeautifulSoup]()
NEW_HTML = dict[str, bs4.BeautifulSoup]()
FAILED = set[str]()


class WebCrawl:
    def __init__(
        self,
        start_url: str,
        scope: list[str],
        header: dict[str, str],
        html_dir: str = "html",
        show_info: bool = True,
    ):
        self.start_url = start_url
        from utils import clean_front, url_in_scope

        self.scope = list(map(clean_front, scope))
        assert url_in_scope(start_url, self.scope)
        self.header = header
        self.html_dir = html_dir
        self.show_info = show_info

    def _get_html(self, target_url: str) -> Union[bs4.BeautifulSoup, None]:
        from utils import url_norm

        target_url = url_norm(target_url)
        if (
            target_url in HTML.keys()
            or target_url in NEW_HTML.keys()
            or target_url in FAILED
            or "login" in target_url
            or "Login" in target_url
            or "user" in target_url
            or "User" in target_url
        ):
            return None
        ok = False
        try_times = 1
        while not ok:
            if try_times > 1:
                print("[INFO] Failed to access {}, retrying...".format(target_url))
            try:
                response = requests.get(target_url, headers=self.header, timeout=1)
            except:
                return None
            if response.status_code == 301 or response.status_code == 302:
                target_url = response.headers["Location"]
            ok = response.ok
            try_times += 1
            if try_times > 5:
                FAILED.add(target_url)
                return None
        try:
            target_html = response.content.decode("utf-8")  # type: ignore
        except:
            FAILED.add(target_url)
            return None
        target_html = bs4.BeautifulSoup(target_html, "html.parser")
        NEW_HTML[target_url] = target_html
        print(
            "Accessed: "
            + target_url
            + ", totally crawled: "
            + str(len(HTML) + len(NEW_HTML))
            + " pages."
        )
        if len(NEW_HTML) == 100:
            self._save_html(self.html_dir)
            HTML.update(NEW_HTML)
            NEW_HTML.clear()
            old_len = len(HTML)
            print("[INFO] Saved HTML in to cache, totally {} pages.".format(old_len))
        return target_html

    def _save_html(
        self,
        dir: str = "html",
    ) -> None:
        for url, html in NEW_HTML.items():
            if url.endswith("/"):
                url = url + "index.htm"
            save_url = url
            if save_url.startswith("https://"):
                save_url = save_url[8:]
            elif save_url.startswith("http://"):
                save_url = save_url[7:]
            if save_url.endswith("/"):
                return
            file_dir = os.path.join(dir, "/".join(save_url.split("/")[:-1]))
            # recursively create dir
            if not os.path.exists(file_dir):
                try:
                    os.makedirs(file_dir)
                except FileExistsError:
                    print("FileExistsError")
                    with open("failure.txt", "a", encoding="utf-8") as f:
                        f.write(url + "\n")
            # check if file already exists
            elif os.path.exists(file_dir + "/" + save_url.split("/")[-1]):
                print(
                    "[INFO] File {} already cached, totally {} files crawled".format(
                        save_url.split("/")[-1], len(NEW_HTML)
                    )
                )
                continue
            # start saving
            try:
                with open(
                    file_dir + "/" + save_url.split("/")[-1], "w", encoding="utf-8"
                ) as f:
                    f.write(html.prettify())
            except FileNotFoundError:
                print("FileNotFoundError")
                with open("failure.txt", "a", encoding="utf-8") as f:
                    f.write(url + "\n")

    def _get_hyperlink(
        self,
        html: bs4.BeautifulSoup,
        base_url: str,
        scope: Optional[list[str]] = None,
    ) -> set[str]:
        from utils import simplifyPath, clean_rear, url_join

        assert base_url
        if not scope:
            scope = self.scope
        all_hyperlinks = set()
        for link in html.find_all("a"):
            hyperlink = link.get("href")
            if hyperlink == "index.html" or hyperlink == "index.htm" or not hyperlink:
                continue
            if "http" not in hyperlink:
                assert base_url
                hyperlink = url_join(base_url, hyperlink)
            hyperlink = simplifyPath(hyperlink)
            if (
                not hyperlink.endswith(".html")
                and not hyperlink.endswith(".htm")
                and not hyperlink.endswith("/")
                or hyperlink in FAILED
                or hyperlink in HTML.keys()
                or hyperlink in NEW_HTML.keys()
            ):
                continue
            comply = False
            for url in scope:
                if url in hyperlink:
                    comply = True
                    break
            if not comply:
                continue
            all_hyperlinks.add(hyperlink)
        return all_hyperlinks

    def _webBFS(self, start_url: str) -> None:
        import time
        from queue import Queue

        url_queue = Queue()
        url_queue.put(start_url)

        while not url_queue.empty():
            target_url = url_queue.get()
            html = self._get_html(target_url)
            if not html:
                continue
            hyperlinks = self._get_hyperlink(html, target_url)
            for hyperlink in hyperlinks:
                if hyperlink.endswith("/"):
                    print(f"Parent: {target_url}, Child: {hyperlink}")
                url_queue.put(hyperlink)
        self._save_html(self.html_dir)
        print("[INFO] Finished, successfully crawled {} pages.".format(len(HTML)))
        return

    def run(self) -> None:
        self._webBFS(self.start_url)
        return


class TextTransformer:
    def __init__(
        self,
        html_database: str,
        cache_dir: str,
        wordsdict_dir: str,
        term_index_dir: str,
        stopwords_dir: Optional[str] = None,
    ):
        self.html_database = html_database
        self.cache_dir = cache_dir
        self.wordsdict_dir = wordsdict_dir
        self.term_index_dir = term_index_dir
        self.stopwords_dir = stopwords_dir

    def load_html(
        self, html_database: Optional[str] = None, cache_dir: Optional[str] = None
    ) -> bool:
        import pickle
        from tqdm import tqdm

        if not cache_dir:
            cache_dir = self.cache_dir
        if cache_dir and os.path.exists(cache_dir + "/HTML.pkl"):
            print("[INFO] Found cached HTML, loading...")
            HTML.update(pickle.load(open(cache_dir + "/HTML.pkl", "rb")))
            print("[INFO] Already loaded, totally {} pages.".format(len(HTML)))
            return True
        if not html_database:
            html_database = self.html_database
        assert os.path.exists(html_database)
        from utils import clean_front

        for root, dirs, files in tqdm(os.walk(html_database), desc="Loading dir: "):
            for file in tqdm(files, desc="Loading files: ", leave=False):
                if not file.endswith(".html") and not file.endswith(".htm"):
                    continue
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    html = bs4.BeautifulSoup(f.read(), "html.parser")
                    cleaned_url = clean_front(os.path.join(root, file))
                    print(cleaned_url)
                    HTML[cleaned_url] = html
        if not HTML:
            return False
        if cache_dir:
            pickle.dump(HTML, open(cache_dir + "/HTML.pkl", "wb"))
        print(
            "[INFO] Process finished, successfully loaded {} pages.".format(len(HTML))
        )
        return True

    def load_html_from_memory(
        self,
        show_info: bool = False,
    ) -> bool:
        HTML.update(NEW_HTML)
        NEW_HTML.clear()
        if len(HTML) == 0:
            return False
        if show_info:
            print("[INFO] Already loaded, totally {} pages.".format(len(HTML)))
        return True

    def load_wordsdict(self, wordsdict_dir: str) -> bool:
        import pickle

        if not os.path.exists(wordsdict_dir + "/wordsdict.pkl"):
            print("[INFO] No cached words dictionary found.")
            return False
        print("[INFO] Found cached dictionary, loading...")
        self.wordsdict = pickle.load(open(wordsdict_dir + "/wordsdict.pkl", "rb"))
        print("[INFO] Totally {} words in dictionary.".format(len(self.wordsdict)))
        return True

    def load_term_index(self, term_index_dir: str) -> bool:
        import pickle

        if not os.path.exists(term_index_dir + "/term_index.pkl"):
            print("[INFO] No cached term index found.")
            return False
        print("[INFO] Found cached term index, loading...")
        self.term_index = pickle.load(open(term_index_dir + "/term_index.pkl", "rb"))
        print("[INFO] Totally {} tokens were detected.".format(len(self.term_index)))
        return True

    def load_stopwords(self, stopwords_dir: Optional[str] = None) -> bool:
        if not stopwords_dir:
            assert self.stopwords_dir
            stopwords_dir = self.stopwords_dir
        assert os.path.exists(stopwords_dir)
        self.stopwords = set()
        with open(stopwords_dir, "r", encoding="utf-8") as f:
            for line in f.readlines():
                self.stopwords.add(line.strip())
        print("[INFO] Totally {} stopwords.".format(len(self.stopwords)))
        return True

    def make_term_index(self, save_dir: str, in_place=True):
        import pickle
        from collections import defaultdict
        import jieba
        from tqdm import tqdm
        from utils import process_html, process_text

        assert HTML
        if self.load_term_index(save_dir):
            return
        self.term_index = list[tuple[str, int]]()
        self.index2url = dict[int, str]()
        self.term_frequency = defaultdict(dict[str, int])
        url_id = 0
        for url, html in tqdm(HTML.items(), desc="Making term-pages pairs: "):
            self.index2url[url_id] = url

            title = process_html(html)[0]
            for token in jieba.cut_for_search(title):
                self.term_index.append((token, url_id))
                self.term_frequency[url_id][token] = (
                    self.term_frequency.get(url_id, dict()).get(token, 0) + 1
                )
            p = html.find_all("p")
            if p:
                for content in p:
                    for token in jieba.cut_for_search(process_text(content.text)):
                        self.term_index.append((token, url_id))
                        self.term_frequency[url_id][token] = (
                            self.term_frequency[url_id].get(token, 0) + 1
                        )
            h1 = html.find_all("h1")
            if h1:
                for content in h1:
                    for token in jieba.cut_for_search(process_text(content.text)):
                        self.term_index.append((token, url_id))
                        self.term_frequency[url_id][token] = (
                            self.term_frequency[url_id].get(token, 0) + 1
                        )
            h2 = html.find_all("h2")
            if h2:
                for content in h2:
                    for token in jieba.cut_for_search(process_text(content.text)):
                        self.term_index.append((token, url_id))
                        self.term_frequency[url_id][token] = (
                            self.term_frequency[url_id].get(token, 0) + 1
                        )
            a = html.find_all("a")
            if a:
                for content in a:
                    for token in jieba.cut_for_search(process_text(content.text)):
                        self.term_index.append((token, url_id))
                        self.term_frequency[url_id][token] = (
                            self.term_frequency[url_id].get(token, 0) + 1
                        )
            mate = html.find_all("mate")
            if mate:
                for content in mate:
                    for token in jieba.cut_for_search(process_text(content.text)):
                        self.term_index.append((token, url_id))
                        self.term_frequency[url_id][token] = (
                            self.term_frequency[url_id].get(token, 0) + 1
                        )
            url_id += 1
        print("[INFO] Totally {} tokens were detected.".format(len(self.term_index)))
        if save_dir:
            pickle.dump(self.index2url, open(save_dir + "/term_index.pkl", "wb"))
        if not in_place:
            return self.term_index
        return None


class IndexMaker:
    def __init__(self, term_index_dir: str):
        self.term_index_dir = term_index_dir

    def load_inverted_index(self, save_dir: str) -> bool:
        import pickle

        if not os.path.exists(save_dir + "/inverted_index.pkl"):
            print("[INFO] No cached inverted index found.")
            return False
        print("[INFO] Found cached inverted index, loading...")
        self.inverted_index = pickle.load(open(save_dir + "/inverted_index.pkl", "rb"))
        print(
            "[INFO] Totally {} tokens in inverted index.".format(
                len(self.inverted_index)
            )
        )
        return True

    def make_inverted_index(
        self,
        save_dir: str,
        term_index: list[tuple[str, int]],
        in_place=True,
    ):
        import pickle
        from collections import defaultdict
        from tqdm import tqdm

        if self.load_inverted_index(save_dir):
            return
        self.inverted_index = defaultdict(set)
        for token, url_id in tqdm(term_index, desc="Making inverted index: "):
            self.inverted_index[token].add(url_id)
        print(
            "[INFO] Totally {} tokens in inverted index.".format(
                len(self.inverted_index)
            )
        )
        if save_dir:
            pickle.dump(
                self.inverted_index, open(save_dir + "/inverted_index.pkl", "wb")
            )
        if not in_place:
            return self.inverted_index
        return None
