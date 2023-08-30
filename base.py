import os
from typing import Optional, Literal


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from index_processor import *
from query_processor import *


class SearchEngine(object):
    def __init__(
        self,
        start_url: list[str],
        scope: list[str],
        html_database: str,
        cache_dir: str,
        stopwords_dir: Optional[str] = None,
    ) -> None:
        self.start_url = start_url
        self.scope = scope
        self.html_database = html_database
        self.cache_dir = cache_dir
        self.stopwords_dir = stopwords_dir

    def crawl(self) -> None:
        from main import HEADERS

        for url in self.start_url:
            robot = WebCrawl(url, self.scope, HEADERS, self.html_database)
            robot.crawl()

    def _make_term_index(self) -> None:
        self.transformer = TextTransformer(
            self.html_database,
            self.cache_dir,
            self.stopwords_dir,
        )
        if not self.transformer._load_term_index_from_memory():
            self.transformer._load_term_index_from_disk(term_index_dir=self.cache_dir)
        self.transformer._load_stopwords(self.stopwords_dir)

    def _make_index(self) -> None:
        self.index_maker = IndexMaker(self.cache_dir)
        self._make_term_index()
        self.index_maker.make_inverted_index("cache", self.transformer.term_index_dict)

    def _using_clean_words(
        self,
        stopwords: Optional[set[str]] = None,
        mode: Literal["current", "learned"] = "current",
    ) -> None:
        assert (
            hasattr(self, "transformer")
            and hasattr(self.transformer, "stopwords")
            and hasattr(self.transformer, "term_frequency")
            and hasattr(self.index_maker, "inverted_index")
        )
        self.transformer._using_clean_words(stopwords, mode)
        self.index_maker._using_clean_words(stopwords, mode)

    def search(
        self,
        user_query,
        top_k=20,
        using_stopwords: bool = False,
        mode: Literal["tf-idf", "LSI", "word2vec"] = "tf-idf",
        page_rank: bool = False,
    ) -> list[str]:
        if not hasattr(self, "transformer") or not hasattr(self, "index_maker"):
            self._make_index()
        self.user_interaction = UserInteraction(user_query, top_k, "and")
        if using_stopwords:
            self.page_ranker = PageRanker(
                self.user_interaction._word_to_term(),
                top_k,
                self.index_maker.inverted_index,
                self.transformer.index2url,
                self.transformer.term_frequency,
                mode=mode,
                using_page_rank=page_rank,
            )
        else:
            self._using_clean_words(self.transformer.stopwords, mode="current")
            self.page_ranker = PageRanker(
                self.user_interaction._word_to_term(),
                top_k,
                self.index_maker.cleaned_inverted_index,
                self.transformer.index2url,
                self.transformer.cleaned_term_frequency,
                mode=mode,
                using_page_rank=page_rank,
            )
        result = self.page_ranker.ranked_page()
        return result
