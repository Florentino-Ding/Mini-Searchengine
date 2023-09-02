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
        mode: Literal["search", "debug"] = "search",
    ) -> None:
        import jieba

        jieba.load_userdict("field_dict/field_dict.txt")

        self.start_url = start_url
        self.scope = scope
        self.html_database = html_database
        self.cache_dir = cache_dir
        self.stopwords_dir = stopwords_dir
        self.tokenizer = jieba.lcut_for_search
        self.mode = mode
        self.tokenizer("warm up")

    def warm_up(self) -> None:
        if not hasattr(self, "transformer") or not hasattr(self, "index_maker"):
            self._prepare_essential()
        self._using_clean_words(self.feature_extractor.stopwords)
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
            self.feature_extractor.prepare_essential(self.mode)  # type: ignore
        if not hasattr(self, "index_maker"):
            self.index_maker = IndexMaker(
                self.cache_dir,
                self.feature_extractor.term_index_dict,
                self.feature_extractor.url2index,
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

    def _using_clean_words(
        self,
        stopwords: Optional[set[str]] = None,
    ) -> None:
        assert (
            hasattr(self, "transformer")
            and hasattr(self.feature_extractor, "stopwords")
            and hasattr(self.feature_extractor, "term_frequency")
            and hasattr(self.index_maker, "inverted_index")
        )
        if not hasattr(self.index_maker, "cleaned_inverted_index"):
            self.index_maker._using_clean_words(stopwords)
        if not hasattr(self.feature_extractor, "cleaned_term_frequency"):
            self.feature_extractor._using_clean_words(stopwords)

    def get_result_urls(
        self,
        user_query,
        top_k=20,
    ) -> list[str]:
        self.user_interaction = UserInteraction(
            self.tokenizer, user_query, top_k, "and"
        )
        self.page_ranker = PageRanker(
            self.user_interaction._word_to_term(),
            top_k,
            self.index_maker.cleaned_inverted_index,
            self.index_maker.cleaned_title_inverted_index,
            self.index_maker.cleaned_anchor_inverted_index,
            self.feature_extractor.index2url,
            self.feature_extractor.cleaned_term_frequency,
            self.feature_extractor.cleaned_anchor_term_frequency,
            self.index_maker.page_rank_score,  # type: ignore
            self.word2vec_model,
        )
        return self.page_ranker.ranked_pages()

    def get_detailed_result(
        self,
        user_query,
        top_k=20,
    ) -> list[dict[str, str]]:
        self.user_interaction = UserInteraction(
            self.tokenizer, user_query, top_k, "and"
        )
        self.page_ranker = PageRanker(
            self.user_interaction._word_to_term(),
            top_k,
            self.index_maker.cleaned_inverted_index,
            self.index_maker.cleaned_title_inverted_index,
            self.index_maker.cleaned_anchor_inverted_index,
            self.feature_extractor.index2url,
            self.feature_extractor.cleaned_term_frequency,
            self.feature_extractor.cleaned_anchor_term_frequency,
            self.index_maker.page_rank_score,  # type: ignore
            self.word2vec_model,
        )
        result = self.page_ranker.ranked_pages(mode="record")
        web_result = list[dict[str, str]]()
        from index_processor import HTML
        from utils import process_text

        for url in result:
            title = HTML[url].find("title")
            if title is None:
                title = "该网页未提供标题"
            else:
                title = process_text(title.get_text())
                if len(title) > 40:
                    title = f"{title[:40]}..."
            summary = process_text(HTML[url].get_text())[:100]
            web_result.append(
                {
                    "url": url,
                    "title": title,
                    "summary": summary,
                }
            )
        return web_result
