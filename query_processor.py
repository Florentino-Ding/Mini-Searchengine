from collections import defaultdict
from typing import Literal
import gensim.models
import torch
from torch import nn
from torch.nn import functional as F


class UserInteraction:
    def __init__(
        self,
        tokenizer,
        top_k: int = 20,
        mode: Literal["and", "or"] = "and",
    ):
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.mode = mode

    def _word_to_term(self, query: str) -> list[str]:
        query_list = query.split(" ")
        term_list = []
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
        index2url: defaultdict[int, dict[str, str]],
        head_term_frequency: dict[int, dict[str, int]],
        text_term_frequency: dict[int, dict[str, int]],
        anchor_term_frequency: dict[int, dict[str, int]],
        page_rank_score: dict[int, float],
        word2vec_model: gensim.models.keyedvectors.Word2VecKeyedVectors,
    ):
        self.query_list = query_set
        self.top_k = top_k
        self.inverted_index = inverted_index
        self.index2url = index2url
        self.head_term_frequency = head_term_frequency
        self.text_term_frequency = text_term_frequency
        self.anchor_term_frequency = anchor_term_frequency
        self.page_rank_score = page_rank_score
        self.word2vec_model = word2vec_model
        self.rank_net = RankNet()
        self.rank_net.init_weights()

    def _appeared_url_id(
        self, terms: list[str], mode: Literal["and", "or"]
    ) -> set[int]:
        result = self.inverted_index[terms[0]]
        for term in terms[1:]:
            if mode == "and":
                result = result & self.inverted_index[term]
            elif mode == "or":
                result = result | self.inverted_index[term]
        return result

    def _custom_tf_score(
        self,
        term: str,
        doc_id: int,
        part: Literal["head", "text", "anchor"],
        normalize: bool = True,
        consider_include: bool = False,
    ) -> float:
        import math

        part_dict = {
            "head": self.head_term_frequency,
            "text": self.text_term_frequency,
            "anchor": self.anchor_term_frequency,
        }
        doc_freq = part_dict[part].get(doc_id, {})
        if term not in doc_freq:
            return 0
        if consider_include:
            appear_time = sum(
                doc_freq[term_in_doc] for term_in_doc in doc_freq if term in term_in_doc
            )
        else:
            appear_time = doc_freq[term]
        if normalize:
            return math.log10(1 + (appear_time / sum(doc_freq.values())))
        else:
            return math.log10(1 + (appear_time))

    def _idf_score(
        self,
        term: str,
        consider_include: bool = False,
    ) -> float:
        import math

        if not consider_include:
            return math.log10(
                len(self.index2url) / (len(self.inverted_index[term]) + 1)
            )
        all_appear_time = sum(
            len(self.inverted_index[t]) for t in self.inverted_index if term in t
        )
        return math.log10(len(self.index2url) / (all_appear_time + 1))

    def _page_rank_score(self, doc_id: int) -> float:
        assert hasattr(self, "page_rank_score")
        return self.page_rank_score[doc_id]

    def _get_url(self, url_id: int) -> str:
        return self.index2url[url_id]["url"]

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
            if similar_terms not in self.inverted_index.keys():
                continue
            else:
                result_query.append(similar_terms)
        return result_query

    def ranked_pages(
        self, mode: Literal["record", "not-record"] = "not-record"
    ) -> list[int]:
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
                self._custom_tf_score(term, page_url_id, "text")
                * self._idf_score(term)
                * math.log10(len(term))
                for term in self.query_list
            )
            raw_tf_idf_score[idx][1] = sum(
                self._custom_tf_score(term, page_url_id, "head")
                * self._idf_score(term)
                * math.log10(len(term))
                for term in self.query_list
            )
            raw_tf_idf_score[idx][2] = sum(
                self._custom_tf_score(term, page_url_id, "anchor")
                * self._idf_score(term)
                * math.log10(len(term))
                for term in self.query_list
            )
            page_rank_score[idx] = self._page_rank_score(page_url_id)
        score = self.rank_net(raw_tf_idf_score, page_rank_score)
        if mode == "record":
            self.last_record = {
                "in": (raw_tf_idf_score, page_rank_score),
                "true_ans": url_ids,
            }
        score_dict = dict(zip(url_ids, score))
        sorted_score = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        return [url_id for url_id, _ in sorted_score[: self.top_k]]

    def user_feedback(self, url_id: int) -> None:
        assert hasattr(self, "last_record")
        assert hasattr(self, "rank_net")

        if not hasattr(self, "tarin_data"):
            self.tranin_data = list[tuple[torch.Tensor, torch.Tensor, int]]()
        self.tranin_data.append(
            (*self.last_record["in"], self.last_record["true_ans"].index(url_id))
        )

    def train(self, epochs: int = 100, lr: float = 0.01) -> None:
        assert hasattr(self, "tranin_data")
        if not hasattr(self, "optimizer"):
            self.optimizer = torch.optim.Adam(self.rank_net.parameters(), lr=lr)
        from tqdm import trange

        for _ in trange(epochs, desc="Training, epoch"):
            for data in self.tranin_data:
                self.rank_net.zero_grad()
                loss = F.cross_entropy(
                    self.rank_net(data[0], data[1]), torch.tensor(data[2])
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


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
        nn.init.constant_(self.tf_idf_sum.weight, 1e-5)
        nn.init.constant_(self.tf_idf_sum_inner.weight, 1)
        nn.init.constant_(self.method_sum.weight, 1e-5)
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
