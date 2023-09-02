from collections import Counter
import os
import re
from typing import Literal, Optional, Union
import requests
import bs4
from bs4 import BeautifulSoup


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
        try_times = 1
        ok = False
        while not ok:
            if try_times > 1:
                print(f"[INFO] Failed to access {target_url}, retrying...")
            try:
                response = requests.get(target_url, headers=self.header, timeout=1)
            except Exception:
                return None
            if response.status_code in {301, 302}:
                target_url = response.headers["Location"]
            ok = response.ok
            try_times += 1
            if try_times > 5:
                FAILED.add(target_url)
                return None
        try:
            target_html = response.content.decode("utf-8")  # type: ignore
        except Exception:
            FAILED.add(target_url)
            return None
        target_html = bs4.BeautifulSoup(target_html, "html.parser")
        NEW_HTML[target_url] = target_html
        print(
            f"Accessed: {target_url}, totally crawled: {str(len(HTML) + len(NEW_HTML))} pages."
        )
        if len(NEW_HTML) == 100:
            self._save_html(self.html_dir)
            HTML.update(NEW_HTML)
            NEW_HTML.clear()
            old_len = len(HTML)
            print(f"[INFO] Saved HTML in to cache, totally {old_len} pages.")
        return target_html

    def _save_html(
        self,
        dir: str = "html",
    ) -> None:
        for url, html in NEW_HTML.items():
            if url.endswith("/"):
                url = f"{url}index.htm"
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
            elif os.path.exists(f"{file_dir}/" + save_url.split("/")[-1]):
                print(
                    f'[INFO] File {save_url.split("/")[-1]} already cached, totally {len(NEW_HTML)} files crawled'
                )
                continue
            # start saving
            try:
                with open(
                    f"{file_dir}/" + save_url.split("/")[-1], "w", encoding="utf-8"
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
            comply = any(url in hyperlink for url in scope)
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
                url_queue.put(hyperlink)
        return

    def crawl(self) -> None:
        self._webBFS(self.start_url)
        self._save_html(self.html_dir)
        print(f"[INFO] Finished, successfully crawled {len(HTML)} pages.")
        return


class FeatureExactor:
    def __init__(
        self,
        html_database: Optional[str],
        cache_dir: str,
        stopwords_dir: Optional[str] = None,
    ):
        self.html_database = html_database
        self.cache_dir = cache_dir
        self.stopwords_dir = stopwords_dir

    def prepare_essential(self, mode: Literal["search", "debug"] = "search") -> None:
        if str(mode) == "debug" and not self._load_html_from_memory():
            self._load_html_cache_from_disk()
        if not self._load_term_index_cache_from_disk(term_index_dir=self.cache_dir):
            self._make_term_index(self.cache_dir)
        self._load_stopwords(self.stopwords_dir)

    def _load_html_cache_from_disk(
        self, html_database: Optional[str] = None, cache_dir: Optional[str] = None
    ) -> bool:
        import pickle

        if not cache_dir:
            cache_dir = self.cache_dir
        assert os.path.exists(cache_dir)
        from main import DEFAULT_CONTINUE

        r = (
            "y"
            if DEFAULT_CONTINUE
            else input("[INFO] Found cached HTML, load it? (y/n)")
        )
        if r in ["y", "Y"]:
            print("[INFO] Loading cached HTML...")
            HTML.update(pickle.load(open(f"{cache_dir}/HTML.pkl", "rb")))
            print(f"[INFO] Already loaded, totally {len(HTML)} pages.")
            return True
        else:
            print("[INFO] Ignored cached HTML, remaking it...")
            return self._make_html_cache(html_database)

    def _make_html_cache(
        self, html_database: Optional[str] = None, cache_dir: Optional[str] = None
    ) -> bool:
        import pickle
        from tqdm import tqdm
        from utils import clean_front

        if not html_database:
            assert self.html_database
            html_database = self.html_database
        if not cache_dir:
            cache_dir = self.cache_dir
        assert os.path.exists(html_database)

        for root, dirs, files in tqdm(os.walk(html_database), desc="Loading dir: "):
            for file in tqdm(files, desc="Loading files: ", leave=False):
                if not file.endswith(".html") and not file.endswith(".htm"):
                    continue
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    html = bs4.BeautifulSoup(f.read(), "html.parser")
                    cleaned_url = clean_front(os.path.join(root, file))
                    HTML[cleaned_url] = html
        if not HTML:
            print("[ERROR] No html file found, please check your html_database.")
            return False
        pickle.dump(HTML, open(f"{cache_dir}/HTML.pkl", "wb"))
        print(f"[INFO] Process finished, successfully loaded {len(HTML)} pages.")
        return True

    def _load_html_from_memory(
        self,
        show_info: bool = True,
    ) -> bool:
        HTML.update(NEW_HTML)
        NEW_HTML.clear()
        if len(HTML) == 0:
            return False
        if show_info:
            print(f"[INFO] Already loaded into memory, totally {len(HTML)} pages.")
        return True

    def _load_word2vec_from_disk(self, cache_dir: str) -> bool:
        import pickle

        if os.path.exists(f"{cache_dir}/word2vec.pkl"):
            self.word2vec_model = pickle.load(open(f"{cache_dir}/word2vec.pkl", "rb"))
            print("[INFO] Load word2vec model from cache.")
            return True
        else:
            print("[INFO] No cached word2vec model found.")
            return False

    def _load_term_index_cache_from_disk(self, term_index_dir: str) -> bool:
        import pickle

        if os.path.exists(f"{term_index_dir}/term_index.pkl"):
            return self._extracted_from__load_term_index_from_disk_8(
                pickle, term_index_dir
            )
        print("[INFO] No cached term index found.")
        return False

    # TODO Rename this here and in `_load_term_index_from_disk`
    def _extracted_from__load_term_index_from_disk_8(self, pickle, term_index_dir):
        if not hasattr(self, "term_index") or not self.term_index_dict:
            self.term_index_dict = pickle.load(
                open(f"{term_index_dir}/term_index.pkl", "rb")
            )
            self.term_index_title, self.term_index, self.term_index_anchor = (
                self.term_index_dict["title"],
                self.term_index_dict["text"],
                self.term_index_dict["anchor"],
            )
        if not hasattr(self, "index2url") or not self.index2url:
            self.index2url = pickle.load(open(f"{term_index_dir}/index2url.pkl", "rb"))
        if not hasattr(self, "url2index") or not self.url2index:
            self.url2index = pickle.load(open(f"{term_index_dir}/url2index.pkl", "rb"))
        if not hasattr(self, "term_frequency") or not self.term_frequency:
            self.term_frequency = pickle.load(
                open(f"{term_index_dir}/term_frequency.pkl", "rb")
            )
        if not hasattr(self, "anchor_term_frequency") or not self.anchor_term_frequency:
            self.anchor_term_frequency = pickle.load(
                open(f"{term_index_dir}/anchor_term_frequency.pkl", "rb")
            )
        print(f"[INFO] Totally {len(self.term_index)} tokens were detected.")
        return True

    def _load_stopwords(self, stopwords_dir: Optional[str] = None) -> bool:
        if not stopwords_dir:
            assert self.stopwords_dir
            stopwords_dir = self.stopwords_dir
        assert os.path.exists(stopwords_dir)
        self.stopwords = set()
        with open(stopwords_dir, "r", encoding="utf-8") as f:
            for line in f:
                self.stopwords.add(line.strip())
        print(f"[INFO] Totally {len(self.stopwords)} stopwords.")
        return True

    def _make_term_index(self, cache_dir: str):
        import pickle
        import jieba
        from tqdm import tqdm
        from utils import process_html, process_text

        self.term_index_title = list[tuple[str, int]]()
        self.index2url = dict[int, str]()
        # when calculating Page-Rank score, we need to know the index of a url
        self.url2index = dict[str, int]()
        for url_id, (url, html) in enumerate(
            tqdm(HTML.items(), desc="Making term-pages pairs: ")
        ):
            self.index2url[url_id] = url
            self.url2index[url] = url_id

            if head := process_html(html)[0]:
                for token in jieba.cut_for_search(head):
                    self.term_index.append((token, url_id))
            if content := TextExtractor().extract(html.prettify()):
                for token in jieba.cut_for_search(content):
                    self.term_index.append((token, url_id))
            if anchor := html.find_all("a"):
                for content in anchor:
                    for token in jieba.cut_for_search(process_text(content.text)):
                        self.term_index.append((token, url_id))
        print(f"[INFO] Totally {+len(self.term_index)} tokens were detected.")
        if cache_dir:
            pickle.dump(self.term_index, open(f"{cache_dir}/term_index.pkl", "wb"))
            pickle.dump(self.index2url, open(f"{cache_dir}/index2url.pkl", "wb"))
            pickle.dump(self.url2index, open(f"{cache_dir}/url2index.pkl", "wb"))
        return None

    def _using_clean_words(
        self,
        stopwords: Optional[set[str]] = None,
    ) -> None:
        import pickle

        if os.path.exists(f"{self.cache_dir}/cleaned_term_frequency.pkl"):
            self.cleaned_term_frequency = pickle.load(
                open(f"{self.cache_dir}/cleaned_term_frequency.pkl", "rb")
            )
        if os.path.exists(f"{self.cache_dir}/cleaned_anchor_term_frequency.pkl"):
            self.cleaned_anchor_term_frequency = pickle.load(
                open(f"{self.cache_dir}/cleaned_anchor_term_frequency.pkl", "rb")
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
            with open(f"{self.cache_dir}/cleaned_term_frequency.pkl", "wb") as f:
                pickle.dump(self.cleaned_term_frequency, f)
        if not hasattr(self, "cleaned_anchor_term_frequency"):
            self.cleaned_anchor_term_frequency = copy.deepcopy(
                self.anchor_term_frequency
            )
            for url_id, term_frequency in self.cleaned_anchor_term_frequency.items():
                for word in stopwords:
                    if word in term_frequency:
                        del term_frequency[word]
            with open(f"{self.cache_dir}/cleaned_anchor_term_frequency.pkl", "wb") as f:
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
        if not os.path.exists(f"{save_dir}/inverted_index.pkl"):
            print("[INFO] No cached inverted index found.")
            ok = False
        else:
            print("[INFO] Found cached inverted index, loading...")
            self.inverted_index = pickle.load(
                open(f"{save_dir}/inverted_index.pkl", "rb")
            )
            print(
                f"[INFO] Totally {len(self.inverted_index)} tokens in inverted index."
            )
        if not os.path.exists(f"{save_dir}/title_inverted_index.pkl"):
            print("[INFO] No cached title inverted index found.")
            ok = False
        else:
            print("[INFO] Found cached title inverted index, loading...")
            self.title_inverted_index = pickle.load(
                open(f"{save_dir}/title_inverted_index.pkl", "rb")
            )
            print(
                f"[INFO] Totally {len(self.title_inverted_index)} tokens in title inverted index."
            )
        if not os.path.exists(f"{save_dir}/anchor_inverted_index.pkl"):
            print("[INFO] No cached anchor inverted index found.")
            ok = False
        else:
            print("[INFO] Found cached anchor inverted index, loading...")
            self.anchor_inverted_index = pickle.load(
                open(f"{save_dir}/anchor_inverted_index.pkl", "rb")
            )
            print(
                f"[INFO] Totally {len(self.anchor_inverted_index)} tokens in anchor inverted index."
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
                if token not in self.inverted_index:
                    self.inverted_index[token] = set()
                self.inverted_index[token].add(url_id)
            pickle.dump(
                self.inverted_index, open(f"{save_dir}/inverted_index.pkl", "wb")
            )
        if not hasattr(self, "title_inverted_index"):
            self.title_inverted_index = dict[str, set[int]]()
            for token, url_id in tqdm(
                term_index_dict["title"], desc="Making title inverted index: "
            ):
                if token not in self.title_inverted_index:
                    self.title_inverted_index[token] = set()
                self.title_inverted_index[token].add(url_id)
            pickle.dump(
                self.title_inverted_index,
                open(f"{save_dir}/title_inverted_index.pkl", "wb"),
            )
        if not hasattr(self, "anchor_inverted_index"):
            self.anchor_inverted_index = dict[str, set[int]]()
            for token, url_id in tqdm(
                term_index_dict["anchor"], desc="Making anchor inverted index: "
            ):
                if token not in self.anchor_inverted_index:
                    self.anchor_inverted_index[token] = set()
                self.anchor_inverted_index[token].add(url_id)
            pickle.dump(
                self.anchor_inverted_index,
                open(f"{save_dir}/anchor_inverted_index.pkl", "wb"),
            )
        print("[INFO] Finish making all inverted index.")
        return None

    def _load_page_rank(self, cache_dir: str) -> bool:
        if not os.path.exists(f"{cache_dir}/page_rank.pkl"):
            print("[INFO] No cached page rank found.")
            return False
        else:
            import pickle

            self.page_rank_score = pickle.load(open(f"{cache_dir}/page_rank.pkl", "rb"))
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
        with open(f"{save_dir}/page_rank.pkl", "wb") as f:
            pickle.dump(self.page_rank_score, f)

    def _using_clean_words(
        self,
        stopwords: Optional[set[str]] = None,
    ) -> None:
        import pickle
        import copy

        if os.path.exists(f"{self.cache_dir}/cleaned_inverted_index.pkl"):
            self.cleaned_inverted_index = pickle.load(
                open(f"{self.cache_dir}/cleaned_inverted_index.pkl", "rb")
            )
        else:
            assert stopwords
            self.cleaned_inverted_index = copy.deepcopy(self.inverted_index)
            for word in stopwords:
                if word in self.cleaned_inverted_index:
                    del self.cleaned_inverted_index[word]
            with open(f"{self.cache_dir}/cleaned_inverted_index.pkl", "wb") as f:
                pickle.dump(self.cleaned_inverted_index, f)
        if os.path.exists(f"{self.cache_dir}/cleaned_title_inverted_index.pkl"):
            self.cleaned_title_inverted_index = pickle.load(
                open(f"{self.cache_dir}/cleaned_title_inverted_index.pkl", "rb")
            )
        else:
            assert stopwords
            self.cleaned_title_inverted_index = copy.deepcopy(self.title_inverted_index)
            for word in stopwords:
                if word in self.cleaned_title_inverted_index:
                    del self.cleaned_title_inverted_index[word]
            with open(f"{self.cache_dir}/cleaned_title_inverted_index.pkl", "wb") as f:
                pickle.dump(self.cleaned_title_inverted_index, f)
        if os.path.exists(f"{self.cache_dir}/cleaned_anchor_inverted_index.pkl"):
            self.cleaned_anchor_inverted_index = pickle.load(
                open(f"{self.cache_dir}/cleaned_anchor_inverted_index.pkl", "rb")
            )
        else:
            assert stopwords
            self.cleaned_anchor_inverted_index = copy.deepcopy(
                self.anchor_inverted_index
            )
            for word in stopwords:
                if word in self.cleaned_anchor_inverted_index:
                    del self.cleaned_anchor_inverted_index[word]
            with open(f"{self.cache_dir}/cleaned_anchor_inverted_index.pkl", "wb") as f:
                pickle.dump(self.cleaned_anchor_inverted_index, f)
        assert (
            hasattr(self, "cleaned_inverted_index")
            and hasattr(self, "cleaned_title_inverted_index")
            and hasattr(self, "cleaned_anchor_inverted_index")
        )


class TextExtractor:
    def _filter_tags(self, html_str, flag):
        """过滤各类标签
        :param html_str: html字符串
        :param flag: 是否移除所有标签
        :return: 过滤标签后的html字符串
        """
        html_str = re.sub("(?is)<!DOCTYPE.*?>", "", html_str)
        html_str = re.sub("(?is)<!--.*?-->", "", html_str)  # remove html comment
        html_str = re.sub(
            "(?is)<script.*?>.*?</script>", "", html_str
        )  # remove javascript
        html_str = re.sub("(?is)<style.*?>.*?</style>", "", html_str)  # remove css
        html_str = re.sub("(?is)<a[\t|\n|\r|\f].*?>.*?</a>", "", html_str)  # remove a
        html_str = re.sub("(?is)<li[^nk].*?>.*?</li>", "", html_str)  # remove li
        # html_str = re.sub('&.{2,5};|&#.{2,5};', '', html_str) #remove special char
        if flag:
            html_str = re.sub("(?is)<.*?>", "", html_str)  # remove tag
        return html_str

    def _extract_text_by_block(self, html_str):
        """根据文本块密度获取正文
        :param html_str: 网页源代码
        :return: 正文文本
        """
        html = self._filter_tags(html_str, True)
        lines = html.split("\n")
        blockwidth = 3
        threshold = 86
        indexDistribution = []
        for i in range(0, len(lines) - blockwidth):
            wordnum = 0
            for j in range(i, i + blockwidth):
                line = re.sub("\\s+", "", lines[j])
                wordnum += len(line)
            indexDistribution.append(wordnum)
        startindex = -1
        endindex = -1
        boolstart = False
        boolend = False
        arcticle_content = []
        for i in range(0, len(indexDistribution) - blockwidth):
            if indexDistribution[i] > threshold and boolstart is False:
                if (
                    indexDistribution[i + 1] != 0
                    or indexDistribution[i + 2] != 0
                    or indexDistribution[i + 3] != 0
                ):
                    boolstart = True
                    startindex = i
                    continue
            if boolstart is True:
                if indexDistribution[i] == 0 or indexDistribution[i + 1] == 0:
                    endindex = i
                    boolend = True
            tmp = []
            if boolend is True:
                for index in range(startindex, endindex + 1):
                    line = lines[index]
                    if len(line.strip()) < 5:
                        continue
                    tmp.append(line.strip() + "\n")
                tmp_str = "".join(tmp)
                if "Copyright" in tmp_str or "版权所有" in tmp_str:
                    continue
                arcticle_content.append(tmp_str)
                boolstart = False
                boolend = False
        return "".join(arcticle_content)

    def _extract_text_by_tag(self, html_str, article):
        """全网页查找根据文本块密度获取的正文的位置，获取文本父级标签内的正文，目的是提高正文准确率
        :param html: 网页html
        :param article: 根据文本块密度获取的正文
        :return: 正文文本
        """
        lines = self._filter_tags(html_str, False)
        soup = BeautifulSoup(lines, "lxml")
        p_list = soup.find_all("p")
        p_in_article = []
        for p in p_list:
            if p.text.strip() in article:
                p_in_article.append(p.parent)
        tuple = Counter(p_in_article).most_common(1)[0]
        article_soup = BeautifulSoup(str(tuple[0]), "xml")
        return self._remove_space(article_soup.text)

    def _remove_space(self, text):
        """移除字符串中的空白字符"""
        text = re.sub("[\t\r\n\f]", "", text)
        return text

    def extract(self, raw_html):
        """抽取正文
        :param url: 网页链接
        :return：正文文本
        """
        if raw_html == None:
            return None
        article_temp = self._extract_text_by_block(raw_html)
        try:
            article = self._extract_text_by_tag(raw_html, article_temp)
        except:
            article = article_temp
        return article
