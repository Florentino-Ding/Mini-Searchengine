import os
from typing import Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


class indexing_processor(object):
    def __init__(self, html_database: Optional[str] = None):
        self.html_database = html_database


class query_processing(object):
    def __init__(self, query):
        self.query = query

    def __repr__(self):
        return str(self.query)
