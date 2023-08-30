from collections import defaultdict
from typing import Literal


class UserInteraction:
    def __init__(
        self,
        query: str,
        top_k: int = 20,
        mode: Literal["and", "or"] = "and",
    ):
        self.query = query
        self.query_list = self._word_to_term()
        self.top_k = top_k
        self.mode = mode

    def _word_to_term(self) -> list[str]:
        import jieba

        if " " in self.query:
            query_list = " ".split(self.query)
        else:
            query_list = [self.query]
        term_list = list()
        for word in query_list:
            term_list.extend(jieba.lcut_for_search(word))
        return term_list

    def get_result(self, result: list[int]):
        raise NotImplementedError


class PageRanker:
    def __init__(
        self,
        query_set: list[str],
        top_k: int,
        inverted_index: dict[str, set[int]],
        index2url: dict[int, str],
        term_frequency: dict[int, dict[str, int]],
        mode: Literal["tf-idf", "LSI", "word2vec"] = "tf-idf",
        using_page_rank: bool = False,
    ):
        self.query_list = query_set
        self.top_k = top_k
        self.inverted_index = inverted_index
        self.index2url = index2url
        self.term_frequency = term_frequency
        self.mode = mode
        self.using_page_rank = using_page_rank

    def _appeared_url_id(
        self, terms: list[str], mode: Literal["and", "or"]
    ) -> set[int]:
        result = self.inverted_index[terms[0]]
        if mode == "and":
            for term in terms[1:]:
                result = result & self.inverted_index[term]
        elif mode == "or":
            for term in terms[1:]:
                result = result | self.inverted_index[term]
        return result

    def _tf_score(
        self,
        term: str,
        doc_id: int,
        normalize: bool = True,
        consider_include: bool = False,
    ) -> float:
        import math

        appear_time = 0
        doc_freq = self.term_frequency[doc_id]
        if consider_include:
            for term_in_doc in doc_freq:
                if term in term_in_doc:
                    appear_time += doc_freq[term_in_doc]
        else:
            appear_time = doc_freq[term]
        if not term in doc_freq:
            return 0
        if normalize:
            return 1 + math.log10(appear_time / sum(doc_freq.values()))
        return 1 + math.log10(appear_time)

    def _idf_score(
        self,
        term: str,
        scope: Literal["pages", "terms"] = "pages",
        consider_include: bool = False,
    ) -> float:
        import math

        return math.log10(len(self.index2url) / (len(self.inverted_index[term]) + 1))

    def _get_url(self, url_id: int) -> str:
        return self.index2url[url_id]

    def ranked_page(self) -> list[str]:
        url_ids = self._appeared_url_id(self.query_list, "and")
        score_dict = dict[int, float]()
        for page_url_id in url_ids:
            score_dict[page_url_id] = sum(
                [
                    self._tf_score(term, page_url_id) * self._idf_score(term)
                    for term in self.query_list
                ]
            )
        sorted_score = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        return [self._get_url(url_id) for url_id, _ in sorted_score[: self.top_k]]
