import os
from typing import Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from index_processor import WebCrawl, TextTransformer, IndexMaker


class IndexProcessor(object):
    def __init__(
        self,
        WebCrawl: WebCrawl,
        TextTransformer: TextTransformer,
        IndexMaker: IndexMaker,
    ):
        self.WebCrawl = WebCrawl
        self.TextTransformer = TextTransformer
        self.IndexMaker = IndexMaker


class QueryProcessor(object):
    def __init__(self, query):
        self.query = query

    def __repr__(self):
        return str(self.query)


class SearchEngine(object):
    def __init__(
        self, index_processor: IndexProcessor, query_processor: QueryProcessor
    ):
        self.index_processor = index_processor
        self.query_processor = query_processor

    def __repr__(self):
        return str(self.index_processor) + str(self.query_processor)
