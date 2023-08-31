import os
from typing import Literal, Optional, Union
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

    def crawl(self) -> None:
        self._webBFS(self.start_url)
        return


class TextTransformer:
    def __init__(
        self,
        html_database: str,
        cache_dir: str,
        stopwords_dir: Optional[str] = None,
    ):
        self.html_database = html_database
        self.cache_dir = cache_dir
        self.stopwords_dir = stopwords_dir

    def prepare_essential(self, mode: Literal["search", "debug"] = "search") -> None:
        if str(mode) == "debug" and not self._load_html_from_memory():
            self._load_html_from_disk()
        if not self._load_term_index_from_memory():
            if not self._load_term_index_from_disk(term_index_dir=self.cache_dir):
                self._make_term_index(self.cache_dir)
        self._load_stopwords(self.stopwords_dir)

    def _load_html_from_disk(
        self, html_database: Optional[str] = None, cache_dir: Optional[str] = None
    ) -> bool:
        import pickle
        from tqdm import tqdm

        if not cache_dir:
            cache_dir = self.cache_dir
        if cache_dir and os.path.exists(cache_dir + "/HTML.pkl") and not HTML:
            from main import DEFAULT_CONTINUE

            r = (
                "y"
                if DEFAULT_CONTINUE
                else input("[INFO] Found cached HTML, load it? (y/n)")
            )
            if r == "y" or r == "Y":
                print("[INFO] Loading cached HTML...")
                HTML.update(pickle.load(open(cache_dir + "/HTML.pkl", "rb")))
                print("[INFO] Already loaded, totally {} pages.".format(len(HTML)))
                return True
            else:
                print("[INFO] Ignored cached HTML, remaking it...")
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
                    HTML[cleaned_url] = html
        if not HTML:
            return False
        if cache_dir:
            pickle.dump(HTML, open(cache_dir + "/HTML.pkl", "wb"))
        print(
            "[INFO] Process finished, successfully loaded {} pages.".format(len(HTML))
        )
        return True

    # def _load_all_html_from_disk(
    #     self,
    #     html_database: str = "all_html",
    #     reflection_path: str = "reflection.json",
    #     cache_dir: Optional[str] = None,
    # ) -> bool:
    #     import pickle
    #     import json
    #     from tqdm import tqdm

    #     from utils import clean_file_format

    #     assert os.path.exists(html_database) and os.path.exists(reflection_path)

    #     if cache_dir and os.path.exists(cache_dir + "/HTML.pkl"):
    #         from main import DEFAULT_CONTINUE

    #         r = (
    #             "y"
    #             if DEFAULT_CONTINUE
    #             else input("[INFO] Found cached HTML, load it? (y/n)")
    #         )
    #         if r == "y" or r == "Y":
    #             print("[INFO] Loading cached HTML...")
    #             HTML.update(pickle.load(open(cache_dir + "/HTML.pkl", "rb")))
    #             print("[INFO] Already loaded, totally {} pages.".format(len(HTML)))
    #             return True
    #         else:
    #             print("[INFO] Ignored cached HTML, remaking it...")
    #     with open(reflection_path, "r", encoding="utf-8") as f:
    #         reflection = json.load(f)
    #         reflection = dict(reflection)
    #     for file in tqdm(os.listdir(html_database), desc="Loading files: "):
    #         with open(os.path.join(html_database, file), "r", encoding="utf-8") as f:
    #             html = bs4.BeautifulSoup(f.read(), "html.parser")
    #             url = reflection[clean_file_format(file)]
    #             HTML[url] = html
    #     if not HTML:
    #         return False
    #     if cache_dir:
    #         print("[INFO] Saving HTML to cache...")
    #         pickle.dump(HTML, open(cache_dir + "/HTML.pkl", "wb"))
    #     print(
    #         "[INFO] Process finished, successfully loaded {} pages.".format(len(HTML))
    #     )
    #     return True

    def _load_html_from_memory(
        self,
        show_info: bool = True,
    ) -> bool:
        HTML.update(NEW_HTML)
        NEW_HTML.clear()
        if len(HTML) == 0:
            return False
        else:
            if show_info:
                print(
                    "[INFO] Already loaded into memory, totally {} pages.".format(
                        len(HTML)
                    )
                )
            return True

    def _load_term_index_from_disk(self, term_index_dir: str) -> bool:
        import pickle

        if not os.path.exists(term_index_dir + "/term_index.pkl"):
            print("[INFO] No cached term index found.")
            return False
        else:
            if not hasattr(self, "term_index") or not self.term_index_dict:
                self.term_index_dict = pickle.load(
                    open(term_index_dir + "/term_index.pkl", "rb")
                )
                self.term_index_title, self.term_index, self.term_index_anchor = (
                    self.term_index_dict["title"],
                    self.term_index_dict["text"],
                    self.term_index_dict["anchor"],
                )
            if not hasattr(self, "index2url") or not self.index2url:
                self.index2url = pickle.load(
                    open(term_index_dir + "/index2url.pkl", "rb")
                )
            if not hasattr(self, "url2index") or not self.url2index:
                self.url2index = pickle.load(
                    open(term_index_dir + "/url2index.pkl", "rb")
                )
            if not hasattr(self, "term_frequency") or not self.term_frequency:
                self.term_frequency = pickle.load(
                    open(term_index_dir + "/term_frequency.pkl", "rb")
                )
            if (
                not hasattr(self, "anchor_term_frequency")
                or not self.anchor_term_frequency
            ):
                self.anchor_term_frequency = pickle.load(
                    open(term_index_dir + "/anchor_term_frequency.pkl", "rb")
                )
            print(
                "[INFO] Totally {} tokens were detected.".format(len(self.term_index))
            )
            return True

    def _load_term_index_from_memory(self) -> bool:
        if (
            not hasattr(self, "term_index")
            or not self.term_index_dict
            or not hasattr(self, "index2url")
            or not self.index2url
            or not hasattr(self, "term_frequency")
            or not self.term_frequency
        ):
            return False
        else:
            print(
                "[INFO] Totally {} tokens were detected in memory.".format(
                    len(self.term_index)
                )
            )
        return True

    def _load_stopwords(self, stopwords_dir: Optional[str] = None) -> bool:
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

    def _make_term_index(self, save_dir: str):
        import pickle
        import jieba
        from tqdm import tqdm
        from utils import process_html, process_text

        if os.path.exists(save_dir + "/term_index.pkl"):
            from main import DEFAULT_CONTINUE

            r = (
                "y"
                if DEFAULT_CONTINUE
                else input("[INFO] Found cached term index, load it? (y/n)")
            )
            if r == "y" or r == "Y":
                print("[INFO] Loading cached term index...")
                self.term_index_dict = pickle.load(
                    open(save_dir + "/term_index.pkl", "rb")
                )
                self.index2url = pickle.load(open(save_dir + "/index2url.pkl", "rb"))
                self.url2index = pickle.load(open(save_dir + "/url2index.pkl", "rb"))
                self.term_frequency = pickle.load(
                    open(save_dir + "/term_frequency.pkl", "rb")
                )
                print(
                    "[INFO] Totally {} tokens were detected.".format(
                        len(self.term_index_dict["text"])
                    )
                )
                return
            else:
                print("[INFO] Ignored cached term index, remaking it...")
        self.term_index = list[tuple[str, int]]()
        self.term_index_title = list[tuple[str, int]]()
        self.term_index_anchor = list[tuple[str, int]]()
        self.index2url = dict[int, str]()
        self.url2index = dict[str, int]()
        self.term_frequency = dict[int, dict[str, int]]()
        self.anchor_term_frequency = dict[int, dict[str, int]]()
        url_id = 0
        for url, html in tqdm(HTML.items(), desc="Making term-pages pairs: "):
            self.index2url[url_id] = url
            self.url2index[url] = url_id

            head = process_html(html)[0]
            if head:
                for token in jieba.cut_for_search(head):
                    self.term_index_title.append((token, url_id))
                    if url_id not in self.term_frequency:
                        self.term_frequency[url_id] = dict()
                    self.term_frequency[url_id][token] = (
                        self.term_frequency[url_id].get(token, 0) + 1
                    )
            content = html.get_text()
            if content:
                for token in jieba.cut_for_search(process_text(content)):
                    self.term_index.append((token, url_id))
                    if url_id not in self.term_frequency:
                        self.term_frequency[url_id] = dict()
                    self.term_frequency[url_id][token] = (
                        self.term_frequency[url_id].get(token, 0) + 1
                    )
            anchor = html.find_all("a")
            if anchor:
                for content in anchor:
                    for token in jieba.cut_for_search(process_text(content.text)):
                        self.term_index_anchor.append((token, url_id))
                        if url_id not in self.anchor_term_frequency:
                            self.anchor_term_frequency[url_id] = dict()
                        self.anchor_term_frequency[url_id][token] = (
                            self.anchor_term_frequency[url_id].get(token, 0) + 1
                        )
            url_id += 1
        print("[INFO] Totally {} tokens were detected.".format(+len(self.term_index)))
        self.term_index_dict = {
            "title": self.term_index_title,
            "text": self.term_index,
            "anchor": self.term_index_anchor,
        }
        if save_dir:
            pickle.dump(self.term_index_dict, open(save_dir + "/term_index.pkl", "wb"))
            pickle.dump(self.index2url, open(save_dir + "/index2url.pkl", "wb"))
            pickle.dump(self.url2index, open(save_dir + "/url2index.pkl", "wb"))
            pickle.dump(
                self.term_frequency, open(save_dir + "/term_frequency.pkl", "wb")
            )
            pickle.dump(
                self.anchor_term_frequency,
                open(save_dir + "/anchor_term_frequency.pkl", "wb"),
            )
        return None

    def _using_clean_words(
        self,
        stopwords: Optional[set[str]] = None,
    ) -> None:
        import pickle

        if os.path.exists(self.cache_dir + "/" + "cleaned_term_frequency.pkl"):
            self.cleaned_term_frequency = pickle.load(
                open(self.cache_dir + "/" + "cleaned_term_frequency.pkl", "rb")
            )
        if os.path.exists(self.cache_dir + "/" + "cleaned_anchor_term_frequency.pkl"):
            self.cleaned_anchor_term_frequency = pickle.load(
                open(self.cache_dir + "/" + "cleaned_anchor_term_frequency.pkl", "rb")
            )
        assert stopwords
        assert hasattr(self, "term_index_dict") and stopwords
        import copy

        if not hasattr(self, "cleaned_term_frequency"):
            self.cleaned_term_frequency = copy.deepcopy(self.term_frequency)
            for url_id, term_frequency in self.cleaned_term_frequency.items():
                for word in stopwords:
                    if word in term_frequency:
                        del term_frequency[word]
            with open(self.cache_dir + "/" + "cleaned_term_frequency.pkl", "wb") as f:
                pickle.dump(self.cleaned_term_frequency, f)
        if not hasattr(self, "cleaned_anchor_term_frequency"):
            self.cleaned_anchor_term_frequency = copy.deepcopy(
                self.anchor_term_frequency
            )
            for url_id, term_frequency in self.cleaned_anchor_term_frequency.items():
                for word in stopwords:
                    if word in term_frequency:
                        del term_frequency[word]
            with open(
                self.cache_dir + "/" + "cleaned_anchor_term_frequency.pkl", "wb"
            ) as f:
                pickle.dump(self.cleaned_anchor_term_frequency, f)


class IndexMaker:
    def __init__(
        self,
        cache_dir: str,
        term_index_dict: dict[str, list[tuple[str, int]]],
        url2index: dict[str, int],
    ):
        self.cache_dir = cache_dir
        self.term_index_dict = term_index_dict
        self.url2index = url2index

    def prepare_essential(self) -> None:
        if not self._load_inverted_index(self.cache_dir):
            self._make_inverted_index(self.cache_dir, self.term_index_dict)
        if not self._load_page_rank(self.cache_dir):
            self.make_page_rank(self.cache_dir)

    def _load_inverted_index(self, save_dir: str) -> bool:
        import pickle

        ok = True
        if not os.path.exists(save_dir + "/inverted_index.pkl"):
            print("[INFO] No cached inverted index found.")
            ok = False
        else:
            print("[INFO] Found cached inverted index, loading...")
            self.inverted_index = pickle.load(
                open(save_dir + "/inverted_index.pkl", "rb")
            )
            print(
                "[INFO] Totally {} tokens in inverted index.".format(
                    len(self.inverted_index)
                )
            )
        if not os.path.exists(save_dir + "/title_inverted_index.pkl"):
            print("[INFO] No cached title inverted index found.")
            ok = False
        else:
            print("[INFO] Found cached title inverted index, loading...")
            self.title_inverted_index = pickle.load(
                open(save_dir + "/title_inverted_index.pkl", "rb")
            )
            print(
                "[INFO] Totally {} tokens in title inverted index.".format(
                    len(self.title_inverted_index)
                )
            )
        if not os.path.exists(save_dir + "/anchor_inverted_index.pkl"):
            print("[INFO] No cached anchor inverted index found.")
            ok = False
        else:
            print("[INFO] Found cached anchor inverted index, loading...")
            self.anchor_inverted_index = pickle.load(
                open(save_dir + "/anchor_inverted_index.pkl", "rb")
            )
            print(
                "[INFO] Totally {} tokens in anchor inverted index.".format(
                    len(self.anchor_inverted_index)
                )
            )
        return ok

    def _make_inverted_index(
        self,
        save_dir: str,
        term_index_dict: dict[str, list[tuple[str, int]]],
    ):
        import pickle
        from tqdm import tqdm

        if not hasattr(self, "inverted_index"):
            self.inverted_index = dict[str, set[int]]()
            for token, url_id in tqdm(
                term_index_dict["text"], desc="Making text inverted index: "
            ):
                if not token in self.inverted_index:
                    self.inverted_index[token] = set()
                self.inverted_index[token].add(url_id)
            pickle.dump(
                self.inverted_index, open(save_dir + "/inverted_index.pkl", "wb")
            )
        if not hasattr(self, "title_inverted_index"):
            self.title_inverted_index = dict[str, set[int]]()
            for token, url_id in tqdm(
                term_index_dict["title"], desc="Making title inverted index: "
            ):
                if not token in self.title_inverted_index:
                    self.title_inverted_index[token] = set()
                self.title_inverted_index[token].add(url_id)
            pickle.dump(
                self.title_inverted_index,
                open(save_dir + "/title_inverted_index.pkl", "wb"),
            )
        if not hasattr(self, "anchor_inverted_index"):
            self.anchor_inverted_index = dict[str, set[int]]()
            for token, url_id in tqdm(
                term_index_dict["anchor"], desc="Making anchor inverted index: "
            ):
                if not token in self.anchor_inverted_index:
                    self.anchor_inverted_index[token] = set()
                self.anchor_inverted_index[token].add(url_id)
            pickle.dump(
                self.anchor_inverted_index,
                open(save_dir + "/anchor_inverted_index.pkl", "wb"),
            )
        print("[INFO] Finish making all inverted index.")
        return None

    def _load_page_rank(self, save_dir: str) -> bool:
        if not os.path.exists(save_dir + "/page_rank.pkl"):
            print("[INFO] No cached page rank found.")
            return False
        else:
            import pickle

            self.page_rank_score = pickle.load(open(save_dir + "/page_rank.pkl", "rb"))
            print("[INFO] Cached page rank loaded.")
            return True

    def make_page_rank(self, save_dir: str) -> None:
        assert HTML and len(HTML) > 0

        import pickle
        import networkx as nx

        self.page_rank_score = dict[int, float]()
        G = nx.DiGraph()
        for url in HTML.keys():
            G.add_node(self.url2index[url])

        # Add edges to the graph
        for url, soup in HTML.items():
            id = self.url2index[url]
            for link in soup.find_all("a"):
                href = link.get("href")
                if href in HTML.keys():
                    G.add_edge(id, self.url2index[href])

        # Calculate pagerank
        self.page_rank_score = nx.pagerank(G)
        print(self.page_rank_score)
        with open(save_dir + "/page_rank.pkl", "wb") as f:
            pickle.dump(self.page_rank_score, f)

    def _using_clean_words(
        self,
        stopwords: Optional[set[str]] = None,
    ) -> None:
        import pickle
        import copy

        if os.path.exists(self.cache_dir + "/" + "cleaned_inverted_index.pkl"):
            self.cleaned_inverted_index = pickle.load(
                open(self.cache_dir + "/" + "cleaned_inverted_index.pkl", "rb")
            )
        else:
            assert stopwords
            self.cleaned_inverted_index = copy.deepcopy(self.inverted_index)
            for word in stopwords:
                if word in self.cleaned_inverted_index:
                    del self.cleaned_inverted_index[word]
            with open(self.cache_dir + "/" + "cleaned_inverted_index.pkl", "wb") as f:
                pickle.dump(self.cleaned_inverted_index, f)
        if os.path.exists(self.cache_dir + "/" + "cleaned_title_inverted_index.pkl"):
            self.cleaned_title_inverted_index = pickle.load(
                open(self.cache_dir + "/" + "cleaned_title_inverted_index.pkl", "rb")
            )
        else:
            assert stopwords
            self.cleaned_title_inverted_index = copy.deepcopy(self.title_inverted_index)
            for word in stopwords:
                if word in self.cleaned_title_inverted_index:
                    del self.cleaned_title_inverted_index[word]
            with open(
                self.cache_dir + "/" + "cleaned_title_inverted_index.pkl", "wb"
            ) as f:
                pickle.dump(self.cleaned_title_inverted_index, f)
        if os.path.exists(self.cache_dir + "/" + "cleaned_anchor_inverted_index.pkl"):
            self.cleaned_anchor_inverted_index = pickle.load(
                open(self.cache_dir + "/" + "cleaned_anchor_inverted_index.pkl", "rb")
            )
        else:
            assert stopwords
            self.cleaned_anchor_inverted_index = copy.deepcopy(
                self.anchor_inverted_index
            )
            for word in stopwords:
                if word in self.cleaned_anchor_inverted_index:
                    del self.cleaned_anchor_inverted_index[word]
            with open(
                self.cache_dir + "/" + "cleaned_anchor_inverted_index.pkl", "wb"
            ) as f:
                pickle.dump(self.cleaned_anchor_inverted_index, f)
        assert (
            hasattr(self, "cleaned_inverted_index")
            and hasattr(self, "cleaned_title_inverted_index")
            and hasattr(self, "cleaned_anchor_inverted_index")
        )
