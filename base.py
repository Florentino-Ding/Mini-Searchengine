import os
import gensim.models.keyedvectors as keyedvectors
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from index_processor import *
from query_processor import *


class SearchEngine(object):
    def __init__(
        self,
        start_url: Optional[str] = None,
        scope: Optional[list[str]] = None,
        html_database: Optional[str] = None,
        cache_dir: str = "cache",
        stopwords_dir: Optional[str] = "stopwords/baidu_stopwords.txt",
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

    def warm_up(self) -> None:
        if not hasattr(self, "transformer") or not hasattr(self, "index_maker"):
            self._prepare_essential()
        self._prepare_word2vec()

    def crawl(self) -> None:
        from main import HEADERS

        assert (
            self.start_url is not None
            and self.scope is not None
            and self.html_database is not None
        )
        for url in self.start_url:
            robot = WebCrawl(url, self.scope, HEADERS, self.html_database)
            robot.crawl()

    def _prepare_essential(self) -> None:
        if not hasattr(self, "feature_extractor"):
            self.feature_extractor = FeatureExactor(
                self.html_database,
                self.cache_dir,
                self.stopwords_dir,
            )
            self.feature_extractor.prepare_essential()  # type: ignore
            if not self.feature_extractor._load_clean_term_index():
                self.feature_extractor._making_clean_words()
        if not hasattr(self, "index_maker"):
            self.index_maker = IndexMaker(
                self.cache_dir,
                self.feature_extractor.all_term_index(),
                self.feature_extractor.all_url2index(),
            )
            self.index_maker.prepare_essential()

    def _prepare_word2vec(self) -> None:
        import pickle

        if os.path.exists(os.path.join(self.cache_dir, "word2vec.pkl")):
            self.word2vec_model = pickle.load(
                open(os.path.join(self.cache_dir, "word2vec.pkl"), "rb")
            )
            print("[INFO] Load word2vec model from cache.")
        else:
            self.word2vec_model = keyedvectors.KeyedVectors.load_word2vec_format(
                os.path.join("word2vec", "sgns.merge.word.bz2"),
                binary=False,
                encoding="utf-8",
                unicode_errors="ignore",
            )
            pickle.dump(
                self.word2vec_model,
                open(os.path.join(self.cache_dir, "word2vec.pkl"), "wb"),
            )
            print("[INFO] Save word2vec model to cache.")

    def get_result_urls(
        self,
        user_query,
        top_k=20,
    ) -> list[str]:
        self.user_interaction = UserInteraction(self.tokenizer, top_k, "and")
        self.page_ranker = PageRanker(
            self.user_interaction._word_to_term(user_query),
            top_k,
            self.index_maker.inverted_index,
            self.feature_extractor.all_index2info(),
            self.feature_extractor.head_term_frequency,
            self.feature_extractor.text_term_frequency,
            self.feature_extractor.anchor_term_frequency,
            self.index_maker.page_rank_score,  # type: ignore
            self.word2vec_model,
        )
        return [
            self.feature_extractor.get_page_info(url_id)["url"]
            for url_id in self.page_ranker.ranked_pages()
        ]

    def get_detailed_result(
        self,
        user_query,
        top_k=20,
    ) -> list[dict[str, str]]:
        self.user_interaction = UserInteraction(self.tokenizer, top_k, "and")
        self.page_ranker = PageRanker(
            self.user_interaction._word_to_term(user_query),
            top_k,
            self.index_maker.inverted_index,
            self.feature_extractor.all_index2info(),
            self.feature_extractor.head_term_frequency,
            self.feature_extractor.text_term_frequency,
            self.feature_extractor.anchor_term_frequency,
            self.index_maker.page_rank_score,  # type: ignore
            self.word2vec_model,
        )
        result = self.page_ranker.ranked_pages(mode="record")
        web_result = list[dict[str, str]]()

        for url_id in result:
            web_result.append(
                {
                    "url": self.feature_extractor.get_page_info(url_id)["url"],
                    "title": self.feature_extractor.get_page_info(url_id)["title"],
                    "summary": self.feature_extractor.get_page_info(url_id)["text"][
                        :100
                    ],
                }
            )
        return web_result
