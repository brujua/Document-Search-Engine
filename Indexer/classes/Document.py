import uuid
from dataclasses import dataclass
from typing import Set,  Dict

from classes import Term


@dataclass
class Document:
    file_name: str
    terms: Dict
    id: int
    size: int
    overhead: int = 0

    def has_term(self, term: Term):
        term_freq = self.terms.get(term, 0)
        self.terms[term] = term_freq + 1

    def get_freq(self, term: Term):
        return self.terms.get(term, 0)

    def get_weight(self, term: Term) -> float:
        if term not in self.terms:
            return 0
        return self.terms[term] * term.get_idf()

    def add_overhead(self, overhead: int):
        self.overhead += overhead

    def __str__(self):
        return "[" + self.file_name + " {" + self.terms.__str__() + "}]"

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if not isinstance(other, Document):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.id == other.id
