from dataclasses import dataclass

@dataclass
class InvertedIndexItem:
    term: str
    doc_freq: int
    offset: int
