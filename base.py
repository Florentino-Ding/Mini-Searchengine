import os
from typing import Optional, Literal
import torch


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
        import jieba

        jieba.load_userdict("field_dict/field_dict.txt")

        self.start_url = start_url
        self.scope = scope
        self.html_database = html_database
        self.cache_dir = cache_dir
        self.stopwords_dir = stopwords_dir
        self.tokenizer = jieba.lcut_for_search
        self.tokenizer("warm up")

    def warm_up_for_search(self) -> None:
        if not hasattr(self, "transformer") or not hasattr(self, "index_maker"):
            self._prepare_essential()

    def crawl(self) -> None:
        from main import HEADERS

        for url in self.start_url:
            robot = WebCrawl(url, self.scope, HEADERS, self.html_database)
            robot.crawl()

    def _prepare_term_index(self) -> None:
        self.transformer = TextTransformer(
            self.html_database,
            self.cache_dir,
            self.stopwords_dir,
        )
        self.transformer.prepare_essential()

    def _prepare_essential(self) -> None:
        if not hasattr(self, "transformer"):
            self._prepare_term_index()
        if not hasattr(self, "index_maker"):
            self.index_maker = IndexMaker(
                self.cache_dir,
                self.transformer.term_index_dict,
                self.transformer.url2index,
            )
            self.index_maker.prepare_essential()

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
        if not hasattr(self.index_maker, "cleaned_inverted_index"):
            self.index_maker._using_clean_words(stopwords, mode)
        if not hasattr(self.transformer, "cleaned_term_frequency"):
            self.transformer._using_clean_words(stopwords, mode)

    def search(
        self,
        user_query,
        top_k=20,
    ) -> list[str]:
        self.user_interaction = UserInteraction(
            self.tokenizer, user_query, top_k, "and"
        )
        self._using_clean_words(self.transformer.stopwords, mode="current")
        self.page_ranker = PageRanker(
            self.user_interaction._word_to_term(),
            top_k,
            self.index_maker.cleaned_inverted_index,
            self.transformer.index2url,
            self.transformer.cleaned_term_frequency,
            self.index_maker.page_rank_score,  # type: ignore
        )
        result = self.page_ranker.ranked_page()
        return result
