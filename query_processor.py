from typing import Literal


class UserInteraction:
    def __init__(
        self,
        tokenizer,
        query: str,
        top_k: int = 20,
        mode: Literal["and", "or"] = "and",
    ):
        self.tokenizer = tokenizer
        self.query = query
        self.query_list = self._word_to_term()
        self.top_k = top_k
        self.mode = mode

    def _word_to_term(self) -> list[str]:
        query_list = self.query.split(" ")
        term_list = list()
        for word in query_list:
            term_list.extend(self.tokenizer(word))
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
        page_rank_score: dict[int, float],
    ):
        self.query_list = query_set
        self.top_k = top_k
        self.inverted_index = inverted_index
        self.index2url = index2url
        self.term_frequency = term_frequency
        self.page_rank_score = page_rank_score

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
        if not term in doc_freq:
            return 0
        if consider_include:
            for term_in_doc in doc_freq:
                if term in term_in_doc:
                    appear_time += doc_freq[term_in_doc]
        else:
            appear_time = doc_freq[term]
        if normalize:
            return 1 + math.log10(appear_time / sum(doc_freq.values()))
        return 1 + math.log10(appear_time)

    def _idf_score(
        self,
        term: str,
        consider_include: bool = False,
    ) -> float:
        import math

        if consider_include:
            all_appear_time = 0
            for t in self.inverted_index:
                if term in t:
                    all_appear_time += len(self.inverted_index[t])
            return math.log10(len(self.index2url) / (all_appear_time + 1))
        else:
            return math.log10(
                len(self.index2url) / (len(self.inverted_index[term]) + 1)
            )

    def _page_rank_score(self, doc_id: int) -> float:
        assert hasattr(self, "page_rank_score")
        return self.page_rank_score[doc_id]

    def _get_url(self, url_id: int) -> str:
        return self.index2url[url_id]

    def ranked_page(
        self,
    ) -> list[str]:
        import math

        url_ids = self._appeared_url_id(self.query_list, "and")
        score_dict = dict[int, float]()
        for page_url_id in url_ids:
            score_dict[page_url_id] = sum(
                [
                    self._tf_score(term, page_url_id)
                    * self._idf_score(term)
                    * math.log10(len(term))
                    for term in self.query_list
                ]
            )
        sorted_score = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        return [self._get_url(url_id) for url_id, _ in sorted_score[: self.top_k]]
