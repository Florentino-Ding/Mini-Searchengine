from typing import Literal
from gensim.models import keyedvectors
import torch
from torch import nn
from torch.nn import functional as F


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
        title_inverted_index: dict[str, set[int]],
        anchor_inverted_index: dict[str, set[int]],
        index2url: dict[int, str],
        term_frequency: dict[int, dict[str, int]],
        anchor_term_frequency: dict[int, dict[str, int]],
        page_rank_score: dict[int, float],
        word2vec_model: keyedvectors.Word2VecKeyedVectors,
    ):
        self.query_list = query_set
        self.top_k = top_k
        self.inverted_index = inverted_index
        self.index2url = index2url
        self.title_inverted_index = title_inverted_index
        self.anchor_inverted_index = anchor_inverted_index
        self.term_frequency = term_frequency
        self.anchor_term_frequency = anchor_term_frequency
        self.page_rank_score = page_rank_score
        self.word2vec_model = word2vec_model
        self.rank_net = RankNet()
        self.rank_net.init_weights()

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

        doc_freq = self.term_frequency[doc_id]
        if not term in doc_freq:
            return 0
        if consider_include:
            appear_time = 0
            for term_in_doc in doc_freq:
                if term in term_in_doc:
                    appear_time += doc_freq[term_in_doc]
        else:
            appear_time = doc_freq[term]
        if normalize:
            return math.log10(1 + (appear_time / sum(doc_freq.values())))
        else:
            return math.log10(1 + (appear_time))

    def _anchor_tf_score(
        self,
        term: str,
        doc_id: int,
        normalize: bool = True,
        consider_include: bool = False,
    ) -> float:
        import math

        anchor_freq = self.anchor_term_frequency.get(doc_id, dict())
        if not anchor_freq:
            return 0
        if not term in anchor_freq:
            return 0
        if consider_include:
            appear_time = 0
            for term_in_anchor in anchor_freq:
                if term in term_in_anchor:
                    appear_time += anchor_freq[term_in_anchor]
        else:
            appear_time = anchor_freq[term]
        if normalize:
            return math.log10(1 + (appear_time / sum(anchor_freq.values())))
        else:
            return math.log10(1 + appear_time)

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

    def _anchor_idf_score(
        self,
        term: str,
        consider_include: bool = False,
    ) -> float:
        import math

        if consider_include:
            all_appear_time = 0
            for t in self.anchor_inverted_index:
                if term in t:
                    all_appear_time += len(self.anchor_inverted_index[t])
            return math.log10(len(self.index2url) / (all_appear_time + 1))
        else:
            return math.log10(
                len(self.index2url)
                / (len(self.anchor_inverted_index.get(term, set())) + 1)
            )

    def _page_rank_score(self, doc_id: int) -> float:
        assert hasattr(self, "page_rank_score")
        return self.page_rank_score[doc_id]

    def _get_url(self, url_id: int) -> str:
        return self.index2url[url_id]

    def _nearby_repleace(self, user_query: list[str]) -> list[str]:
        import math

        result_query = list[str]()
        for term in user_query:
            if term in self.inverted_index.keys():
                result_query.append(term)
                continue
            if term not in self.word2vec_model:
                continue
            similar_terms, _ = self.word2vec_model.most_similar(term)[0]
            if not similar_terms in self.inverted_index.keys():
                continue
            else:
                result_query.append(similar_terms)
        return result_query

    def ranked_page(
        self,
    ) -> list[str]:
        import math
        from base import DEVICE

        self.query_list = self._nearby_repleace(self.query_list)
        url_ids = self._appeared_url_id(self.query_list, "and")
        score_dict = dict[int, float]()
        raw_tf_idf_score = torch.empty((len(url_ids), 3), device=DEVICE)
        page_rank_score = torch.empty((len(url_ids), 1), device=DEVICE)
        url_ids = list(url_ids)
        for idx, page_url_id in enumerate(url_ids):
            raw_tf_idf_score[idx][0] = sum(
                [
                    self._tf_score(term, page_url_id) * self._idf_score(term)
                    for term in self.query_list
                ]
            )
            raw_tf_idf_score[idx][1] = sum(
                [
                    math.exp(
                        int(page_url_id in self.title_inverted_index.get(term, []))
                    )
                    for term in self.query_list
                ]
            )
            raw_tf_idf_score[idx][2] = sum(
                [
                    self._anchor_tf_score(term, page_url_id)
                    * self._anchor_idf_score(term)
                    for term in self.query_list
                ]
            )
            page_rank_score[idx] = self._page_rank_score(page_url_id)
        score = self.rank_net(raw_tf_idf_score, page_rank_score)
        score_dict = {url_id: s for url_id, s in zip(url_ids, score)}
        sorted_score = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        return [self._get_url(url_id) for url_id, _ in sorted_score[: self.top_k]]


class RankNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.tf_idf_sum_inner = nn.Linear(3, 3, bias=False)
        self.tf_idf_sum = nn.Linear(3, 1, bias=False)
        self.activation1 = F.relu
        self.method_sum_inner = nn.Linear(2, 2, bias=False)
        self.method_sum = nn.Linear(2, 1, bias=False)
        self.activation2 = F.relu
        self.softmax = F.softmax

    def init_weights(self):
        nn.init.constant_(self.tf_idf_sum.weight, 0)
        nn.init.constant_(self.tf_idf_sum_inner.weight, 1)
        nn.init.constant_(self.method_sum.weight, 0)
        nn.init.constant_(self.method_sum_inner.weight, 1)

    def forward(self, raw_tf_idf_score, page_rank_score) -> torch.Tensor:
        tf_idf_score = self.activation1(self.tf_idf_sum_inner(raw_tf_idf_score))
        tf_idf_score = self.activation1(
            self.tf_idf_sum(tf_idf_score).reshape(-1) + raw_tf_idf_score[:, 0]
        )
        tf_idf_score = torch.unsqueeze(tf_idf_score, dim=1)
        methods_score = torch.cat([tf_idf_score, page_rank_score], dim=1)
        score = self.activation2(self.method_sum_inner(methods_score))
        score = self.activation2(
            self.method_sum(score) + torch.unsqueeze(methods_score[:, 0], dim=1)
        )
        score = torch.squeeze(score, dim=1) * 1000
        return self.softmax(score, dim=0)
