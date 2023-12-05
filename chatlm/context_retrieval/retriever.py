from dataclasses import dataclass
from typing import List

@dataclass
class Doc:
    text: str

@dataclass
class ContextRetriever:
    name: str
    index_name: str

    def similarity_search(self, query, k=10, method:str="cosine") -> List[str]:
        docs = ["Promise"]*k
        return [Doc(doc) for doc in docs]

    def __call__(self, query):
        return self.similarity_search(query)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass