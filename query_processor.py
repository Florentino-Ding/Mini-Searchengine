from collections import defaultdict
from typing import Literal


class UserInteraction:
    def __init__(
        self,
        query: str,
        inverted_index: defaultdict[str, set],
        index2url: dict[int, str],
        term_frequency: dict[int, dict[str, int]],
        mode: Literal["and", "or"] = "and",
    ):
        self.query = query
        self.inverted_index = inverted_index
        self.index2url = index2url
        self.term_frequency = term_frequency
        self.mode = mode

    def _word_to_term(self) -> set[str]:
        import jieba

        return set(jieba.lcut_for_search(self.query))

    def _get_url(self, doc_id: int) -> str:
        return self.index2url[doc_id]

    def _tf_score(self, term: str, doc_id: int) -> float:
        import math

        doc_freq = self.term_frequency[doc_id]
        if not term in doc_freq:
            return 0
        return 1 + math.log10(doc_freq[term])

    def _idf_score(self, term: str) -> float:
        import math

        return math.log10(len(self.index2url) / len(self.inverted_index[term]))
