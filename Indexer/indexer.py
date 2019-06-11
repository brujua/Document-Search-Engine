import os
import sys
import struct
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Set
from classes.InvertedIndexItem import InvertedIndexItem

from classes.Document import Document
from classes.Term import Term
from tokenizer.tokenizer import tokenizar, TOKEN_MAX_LEN
from tokenizer.tokenizer import sacar_palabras_vacias
from utils import get_files

ERROR_ARGS = "Invalid Arguments. \nUsage: python " + sys.argv[0] + " <corpus-dir> <stop-words-file>"
EXIT_QUERY = "--exit"
ERROR_WRONG_QUERY_SYNTAX = "Query given has invalid syntax"
INDEX_FILE_NAME = "index.bin"

QUERY_OPERANDS = {"and": "and", "or": "or", "not": "not"}  # To support query operands in other languages: change values
POSTING_FORMAT = "2I"
POSTING_SIZE = 8  # bytes
VOCABULARY_ITEM_SIZE = TOKEN_MAX_LEN + 8  # bytes (token_max_len characters each 1 byte, 4 bytes for doc_freq,
# 4 bytes for offset)
MAX_RESULTS = 20

documents = []  # List[Document]
terms_doc_dic = {}  # { term_name : Term, }
vocabulary = {}  # { term_name : InvertedIndexItem, }


def main(*args):
    empty_words = get_empty_words(args[1])
    files = get_files(args[0])
    index(files, empty_words)
    save_index_to_disk()
    # show_overhead_stats()
    while True:
        print("Enter Query: (or type", EXIT_QUERY, " )")
        query = input()
        if query == EXIT_QUERY:
            break
        try:
            result_docs = resolve_query(query, empty_words)
            print("Documents: ")
            for doc in result_docs[:MAX_RESULTS]:
                print(doc.file_name)
        except SyntaxError as err:
            print(err)


def index(files: List[str], empty_words: List[str]):
    print("indexing...")
    progress_bar = tqdm(total=len(files), unit="file")
    for file_name in files:
        doc = Document(file_name, {}, len(documents), os.path.getsize(file_name))
        documents.append(doc)
        with open(file_name, encoding="utf-8", errors="ignore") as file:
            tokens = tokenizar(file.read())
        tokens = sacar_palabras_vacias(tokens, empty_words)
        for token in tokens:
            term = terms_doc_dic.get(token, Term(token, set(), len(terms_doc_dic)))
            terms_doc_dic[token] = term
            doc.has_term(term)
            term.found_in(doc)
        progress_bar.update(1)
    progress_bar.close()


def save_index_to_disk():
    offset = 0
    with open(INDEX_FILE_NAME, "wb") as file:
        for term in terms_doc_dic.values():
            vocabulary[term.name] = InvertedIndexItem(term.name, term.doc_freq, offset)
            for doc in term.documents:
                documents[doc.id].add_overhead(POSTING_SIZE)
                values = (doc.id, doc.get_freq(term))
                packed_data = struct.pack(POSTING_FORMAT, *values)
                file.write(packed_data)
                offset += POSTING_SIZE


def resolve_query(query: str, empty_words: List[str] = []) -> List[Document]:
    # remove empty words that are not "and", "or" or "not"
    words = [w for w in tokenizar(query) if (w not in empty_words) or (w in QUERY_OPERANDS.values())]
    docs = []
    and_f = False
    or_f = False
    not_f = False
    for i, word in enumerate(words):
        if word == QUERY_OPERANDS["not"]:
            not_f = True
        elif word == QUERY_OPERANDS["or"]:
            or_f = True
        elif word == QUERY_OPERANDS["and"]:
            and_f = True
        else:
            if i == 0:  # first word
                docs = retrieve_docs(word)
            elif and_f:
                docs = apply_and(docs, retrieve_docs(word))
                and_f = False
            elif or_f:
                docs = apply_or(docs, retrieve_docs(word))
                or_f = False
            elif not_f:
                docs = apply_not(docs, retrieve_docs(word))
            else:
                raise SyntaxError(ERROR_WRONG_QUERY_SYNTAX)
    return docs


def resolve_query_in_ram(query: str) -> Set[Document]:
    # remove empty words that are not "and", "or" or "not"
    words = tokenizar(query)
    docs = set()
    and_f = False
    or_f = False
    not_f = False
    for i, word in enumerate(words):
        if word == QUERY_OPERANDS["not"]:
            not_f = True
        elif word == QUERY_OPERANDS["or"]:
            or_f = True
        elif word == QUERY_OPERANDS["and"]:
            and_f = True
        else:
            if i == 0:  # first word
                docs = retrieve_docs_ram(word)
            elif and_f:
                docs = apply_and(docs, retrieve_docs_ram(word))
                and_f = False
            elif or_f:
                docs = apply_or(docs, retrieve_docs_ram(word))
                or_f = False
            elif not_f:
                docs = apply_not(docs, retrieve_docs_ram(word))
            else:
                raise SyntaxError(ERROR_WRONG_QUERY_SYNTAX)
    return docs


def retrieve_docs_ram(word: str) -> Set[Document]:
    docs = set()
    term = terms_doc_dic.get(word)
    if term is not None:
        docs = term.documents
    return docs


def apply_and(docs: Set[Document], new_docs: Set[Document]) -> Set[Document]:
    return docs.intersection(new_docs)


def apply_or(docs: Set[Document], new_docs: Set[Document]) -> Set[Document]:
    return docs.union(new_docs)


def apply_not(docs: Set[Document], new_docs: Set[Document]) -> Set[Document]:
    return docs.difference(new_docs)


def retrieve_docs(word: str) -> Set[Document]:
    posting_parts = 2  # the postings are (doc_id, tf)
    docs = set()
    term = vocabulary.get(word)
    if term is not None:
        struct_format = '{}I'.format(term.doc_freq * posting_parts)
        with open(INDEX_FILE_NAME, "rb") as file:
            file.seek(term.offset)
            content = file.read(POSTING_SIZE * term.doc_freq)
            unpacked_data = struct.unpack(struct_format, content)
        for doc_id in unpacked_data[::posting_parts]:
            docs.add(documents[doc_id])
    return docs


def show_overhead_stats():
    total_overhead = 0
    posting_list_sizes = []
    for term in terms_doc_dic.values():
        posting_list_sizes.append(term.doc_freq * POSTING_SIZE)
        total_overhead += term.doc_freq * POSTING_SIZE + VOCABULARY_ITEM_SIZE
    # Plotting posting lists sizes distribution
    plt.xlabel('Bytes')
    plt.ylabel('Number of posting lists')
    plt.title('Posting list size distribution')
    plt.hist(posting_list_sizes, bins=200)
    plt.show()

    # calculating overhead per document
    coll_size = 0
    doc_overheads = []
    for doc in documents:
        doc_overheads.append(doc.overhead/doc.size)
        coll_size += doc.size

    # Plotting overhead per document distribution
    plt.xlabel('Overhead/size in Bytes')
    plt.ylabel('Number of documents')
    plt.title('File')
    plt.hist(doc_overheads)
    plt.show()

    print("Total Overhead:", total_overhead, "bytes", "from a collection of ", coll_size, "bytes total")


def get_empty_words(file_name: str):
    with open(file_name, encoding="utf-8") as file:
        return tokenizar(file.read())


if __name__ == '__main__':
    if len(sys.argv) is not 3:
        print(ERROR_ARGS)
    else:
        main(*sys.argv[1:])
