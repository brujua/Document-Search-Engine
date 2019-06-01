import uuid
from dataclasses import dataclass
from typing import Set, List

from classes import Document


@dataclass
class Term:
    name: str
    documents: Set
    id: int
    corpus_freq: int = 0
    doc_freq: int = 0
    idf: float = -1

    def found_in(self, doc: Document):
        if doc not in self.documents:
            self.documents.add(doc)
            self.doc_freq += 1
        self.corpus_freq += 1

    def set_idf(self, idf: float):
        self.idf = idf

    def get_idf(self):
        return self.idf

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.id
