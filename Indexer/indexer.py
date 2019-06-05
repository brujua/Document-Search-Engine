import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List
from math import log2
from classes.InvertedIndexItem import InvertedIndexItem

from classes.Document import Document
from classes.Term import Term
from tokenizer.tokenizer import tokenizar
from tokenizer.tokenizer import sacar_palabras_vacias
from utils import get_files

ERROR_ARGS = "Invalid Arguments. \nUsage: python " + sys.argv[0] + " <corpus-dir> <stop-words-file>"
EXIT_QUERY = "--exit"
MAX_RESULTS = 10
corpus_terms = {}
documents = []
inverted_index = {}  # [InvertedIndexItem]
INDEX_FILE_NAME = "index.bin"
POSTING_FORMAT = "2I"
POSTING_SIZE = 8  # bytes


def main(*args):
    empty_words = get_empty_words(args[1])
    files = get_files(args[0])
    index(files, empty_words)
    save_index_to_disk()
    calculate_overhead_stats()


def index(files: List[str], empty_words: List[str]):
    print("indexing...")
    progress_bar = tqdm(total=len(files), unit="file")
    for file_name in files:
        doc = Document(file_name, {}, len(documents))
        documents.append(doc)
        with open(file_name, encoding="utf-8", errors="ignore") as file:
            tokens = tokenizar(file.read())
        tokens = sacar_palabras_vacias(tokens, empty_words)
        for token in tokens:
            term = corpus_terms.get(token, Term(token, set(), len(corpus_terms)))
            corpus_terms[token] = term
            doc.has_term(term)
            term.found_in(doc)
        progress_bar.update(1)
    calculate_idfs()
    progress_bar.close()


def save_index_to_disk():
    offset = 0
    with open(INDEX_FILE_NAME, "wb") as file:
        for term in corpus_terms.values():
            inverted_index[term.name] = InvertedIndexItem(term.name, term.doc_freq, offset)
            for doc in term.documents:
                documents[doc.id].add_overhead(POSTING_SIZE)
                values = (doc.id, doc.get_freq(term))
                packed_data = struct.pack(POSTING_FORMAT, *values)
                file.write(packed_data)
                offset += POSTING_SIZE


def calculate_overhead_stats():
    overheads = []
    for doc in documents:
        overheads.append(doc.overhead)
    plt.xlabel('Bytes')
    plt.ylabel('Number of posting lists')
    plt.title('Positing list size distribution')
    plt.hist(overheads, bins=50)
    plt.show()


def retrieve_docs(term: InvertedIndexItem):
    docs = []
    with open(INDEX_FILE_NAME, "rb") as file:
        for doc_number in range(0, term.doc_freq):
            file.seek(term.offset + (POSTING_SIZE * doc_number))
            content = file.read(POSTING_SIZE)
            docs.append(struct.unpack(POSTING_FORMAT, content))
    return docs


def calculate_similitude(query_vector: List[float], doc_vector: List[float]):
    #  Angle between vectors = D . Q / |D| * |Q|
    return np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))


def calculate_idfs():
    for term in corpus_terms.values():
        term.set_idf(log2(len(documents) / term.doc_freq))


def get_empty_words(file_name: str):
    with open(file_name, encoding="utf-8") as file:
        return tokenizar(file.read())


if __name__ == '__main__':
    if len(sys.argv) is not 3:
        print(ERROR_ARGS)
    else:
        main(*sys.argv[1:])
