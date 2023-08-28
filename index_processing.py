import os
from typing import Optional, Union
import requests
import bs4
import chardet

from base import indexing_processor

HTML = dict[str, bs4.BeautifulSoup]()
NEW_HTML = dict[str, bs4.BeautifulSoup]()
FAILED = set[str]()


class crawl(indexing_processor):
    def __init__(
        self,
        start_url: str,
        header: dict[str, str],
        max_depth: int = 5,
        max_threads: int = 5,
        html_dir: str = "html",
        show_info: bool = True,
        independent_html: bool = False,
    ):
        super().__init__(html_dir)
        self.max_depth = max_depth
        self.start_url = start_url
        self.header = header
        self.max_threads = max_threads
        self.html_dir = html_dir
        self.show_info = show_info

    def _get_html(
        self, target_url: str, save_frequency: int = 100
    ) -> Union[bs4.BeautifulSoup, None]:
        if target_url in HTML.keys() or target_url in NEW_HTML.keys():
            return None
        try:
            response = requests.get(target_url, headers=self.header)
        except:
            FAILED.add(target_url)
            return None
        try_times = 1
        while not response.ok:
            if response.status_code == 301 or response.status_code == 302:
                target_url = response.headers["Location"]
            try:
                response = requests.get(target_url, headers=self.header)
            except:
                FAILED.add(target_url)
                return None
            try_times += 1
            if try_times > 5:
                FAILED.add(target_url)
                return None
        coding = chardet.detect(response.content)["encoding"]
        try:
            target_html = response.content.decode(coding)  # type: ignore
        except:
            return None
        target_html = bs4.BeautifulSoup(target_html, "html.parser")
        NEW_HTML[target_url] = target_html
        print(
            "Accessed: "
            + target_url
            + ", totally crawled: "
            + str(len(NEW_HTML) + len(HTML))
            + " pages."
        )
        if len(NEW_HTML) == save_frequency:
            self._save_html("html")
            HTML.update(NEW_HTML)
            NEW_HTML.clear()
        return target_html

    def _save_html(
        self,
        dir: str = "html",
    ) -> None:
        for url, html in NEW_HTML.items():
            save_url = url
            if save_url.startswith("https://"):
                save_url = save_url[8:]
            elif save_url.startswith("http://"):
                save_url = save_url[7:]
            if save_url.endswith("/"):
                save_url = save_url + "index.html"
            file_dir = os.path.join(dir, "/".join(save_url.split("/")[:-1]))
            if not os.path.exists(file_dir):
                try:
                    os.makedirs(file_dir)
                except FileExistsError:
                    print("FileExistsError")
                    with open("failure.txt", "a", encoding="utf-8") as f:
                        f.write(url + "\n")
            elif os.path.exists(file_dir + "/" + save_url.split("/")[-1]):
                print(
                    "[INFO] File {} already cached, totally {} files crawled".format(
                        save_url.split("/")[-1], len(NEW_HTML)
                    )
                )
                continue
            try:
                with open(
                    file_dir + "/" + save_url.split("/")[-1], "w", encoding="utf-8"
                ) as f:
                    f.write(html.prettify())
            except FileNotFoundError:
                print("FileNotFoundError")
                with open("failure.txt", "a", encoding="utf-8") as f:
                    f.write(url + "\n")
            if self.show_info:
                print(
                    "[INFO] Totally {} pages have been crawled.".format(len(NEW_HTML))
                )

    def _get_hyperlink(self, html: bs4.BeautifulSoup, base_url: str) -> set[str]:
        from utils import simplifyPath, clean_rear, url_join

        assert base_url
        all_hyperlinks = set()
        for link in html.find_all("a"):
            hyperlink = link.get("href")
            if not hyperlink or hyperlink == "index.html" or hyperlink == "index.htm":
                continue
            if "http" not in hyperlink:
                assert base_url
                hyperlink = url_join(base_url, hyperlink)
            elif "ruc.edu.cn" not in hyperlink or "login" in hyperlink:
                continue
            else:
                hyperlink = clean_rear(hyperlink)
            if (
                not hyperlink.endswith(".html")
                and not hyperlink.endswith(".htm")
                and not hyperlink.endswith("/")
            ):
                continue
            all_hyperlinks.add(simplifyPath(hyperlink))
        return all_hyperlinks

    def _webDFS(self, start_url: str, depth: int) -> None:
        import threading
        from utils import url_norm

        thread_lock = threading.Lock()

        if depth == 0:
            return
        start_url = url_norm(start_url)
        thread_lock.acquire()
        html = self._get_html(start_url)
        thread_lock.release()
        if not html:
            return
        hyperlinks = self._get_hyperlink(html, start_url)
        hyperlinks = hyperlinks - set(HTML.keys())
        if self.max_threads > threading.active_count() and self.max_threads == 1:
            assert self.max_threads > 0
            threads = list()
            for url in hyperlinks:
                thread = threading.Thread(target=self._webDFS, args=(url, depth - 1))
                threads.append(thread)
                if len(threads) == self.max_threads:
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join()
                    threads.clear()
        else:
            for url in hyperlinks:
                self._webDFS(url, depth - 1)
        return

    def run(self) -> None:
        self._webDFS(self.start_url, self.max_depth)
        return


class text_transformer(indexing_processor):
    def __init__(self, html_database: str):
        super().__init__(html_database)
        self.html_database = html_database

    def load_html(
        self, html_database: Optional[str] = None, cache_dir: Optional[str] = None
    ) -> bool:
        import pickle
        from tqdm import tqdm

        if cache_dir and os.path.exists(cache_dir):
            print("[INFO] Found cached HTML, loading...")
            HTML.update(pickle.load(open(cache_dir, "rb")))
            print("[INFO] Already loaded, totally {} pages.".format(len(HTML)))
            return True
        if not html_database:
            html_database = self.html_database
        assert os.path.exists(html_database)
        from utils import clean_rear, clean_front

        for root, dirs, files in tqdm(os.walk(html_database), desc="Loading dir: "):
            for file in tqdm(files, desc="Loading files: ", leave=False):
                if not file.endswith(".html") and not file.endswith(".htm"):
                    continue
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    html = bs4.BeautifulSoup(f.read(), "html.parser")
                    cleaned_url = clean_rear(clean_front(os.path.join(root, file)))
                    HTML[cleaned_url] = html
        if cache_dir:
            pickle.dump(HTML, open(cache_dir, "wb"))
        print(
            "[INFO] Process finished, successfully loaded {} pages.".format(len(HTML))
        )
        return True

    def load_from_memory(
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

    def make_dict(self, save_dir: str):
        import pickle
        import jieba
        from tqdm import tqdm
        from utils import process_html

        assert HTML
        if os.path.exists(save_dir):
            print("[INFO] Found cached dictionary, loading...")
            self.word_dict = pickle.load(open(save_dir, "rb"))
            print("[INFO] Found cached dictionary.")
            return
        self.word_dict = set()
        for html in tqdm(HTML.values(), desc="Making dictionary: "):
            title = process_html(html)[0]
            self.word_dict.update(jieba.lcut_for_search(title))
            for content in html.find_all("p"):
                self.word_dict.update(jieba.lcut_for_search(content.text))
            for content in html.find_all("h1"):
                self.word_dict.update(jieba.lcut_for_search(content.text))
            for content in html.find_all("h2"):
                self.word_dict.update(jieba.lcut_for_search(content.text))
            for content in html.find_all("a"):
                self.word_dict.update(jieba.lcut_for_search(content.text))
            for content in html.find_all("mate"):
                self.word_dict.update(jieba.lcut_for_search(content.text))
        print("[INFO] Totally {} words in dictionary.".format(len(self.word_dict)))
        if save_dir:
            pickle.dump(self.word_dict, open(save_dir, "wb"))


class index_creator(indexing_processor):
    def __init__(self, index):
        super().__init__(index)
