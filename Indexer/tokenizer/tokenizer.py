from typing import List

import unidecode
import re
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer

language = "spanish"
TERMS_FILE_NAME = "terminos.txt"
STATS_FILE_NAME = "estadisticas.txt"
FREQ_FILE_NAME = "frecuencias.txt"
TOKEN_MAX_LEN = 30
TOKEN_MIN_LEN = 2
SPACE = " "
NUMBER_REGEX = r"(?<![a-zA-Z])\d+((\.|,)\d+)?(?![a-zA-Z])"  # r"( |\n)+[0-9]+"
ACRONYMS_ABBREVIATIONS_REGEX = r"([A-Z]([a-zA-Z]*)\.)+"
MAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
#  from a response in https://stackoverflow.com/questions/6718633/python-regular-expression-again-match-url
URL_REGEX = r"(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:.,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"
NAME_REGEX = r"[A-Z][a-záéóúí]+( [A-Z][a-záéóúí]+)+"
SPECIAL_CHARS = [
    ".",
    "!",
    "="
    "+",
    "-",
    "/",
    ",",
    ":",
    ";",
    "'",
    "{",
    "}",
    "[",
    "]",
    "?",
    "¿",
    ">",
    "<",
    "/",
    '\\',
    "|",
    "\"",
    "\'",
    "_",
    "`",
    "~",
    "@",
    "#",
    "$",
    "%",
    "^",
    "&",
    "*",
    "(",
    ")",
    "\n",
    "\t"
]

stemmer_snowball = SnowballStemmer(language)
stemmer_porter = PorterStemmer()
stemmer_lancaster = LancasterStemmer()
stemmers = {
        "snowball": stemmer_snowball,
        "porter": stemmer_porter,
        "lancaster": stemmer_lancaster
    }


def tokenizar(text: str):
    text = text.casefold()
    for char in SPECIAL_CHARS:
        text = text.replace(char, SPACE)
    text = unidecode.unidecode(text)  # remove accents
    text = re.sub(NUMBER_REGEX, SPACE, text)  # remove numbers
    return [token for token in text.split() if TOKEN_MIN_LEN <= len(token) <= TOKEN_MAX_LEN]


def tokenizar_con_reglas(text: str) -> List[str]:
    tokens = []
    abreviaturas = get_abreviaturas(text)
    mails_y_urls = get_mails_and_urls(text)
    numeros = get_numeros(text)
    nombres = get_names(text)

    tokens.extend(tokenizar(text))
    tokens.extend(abreviaturas)
    tokens.extend(mails_y_urls)
    tokens.extend(numeros)
    tokens.extend(nombres)
    return tokens


def tokenizar_con_stemming(text: str, stemm_algorithm: str = "snowball") -> List[str]:
    """
    Returns the list of tokens contained in the provided text, using the stemmer specified
    :param text: string to tokenize
    :param stemm_algorithm: string with possible values "snowball", "porter", "lancaster".
    :return:
    """
    stemmer = stemmers.get(stemm_algorithm, stemmer_lancaster)
    return [stemmer.stem(token) for token in tokenizar(text)]


def sacar_palabras_vacias(lista_tokens: List[str], lista_vacias: List[str]):
    return [token for token in lista_tokens if token not in lista_vacias]


def get_abreviaturas(string: str) -> List[str]:
    return _extract_by_rule(string, ACRONYMS_ABBREVIATIONS_REGEX)


def get_mails_and_urls(string: str) -> List[str]:
    tokens = _extract_by_rule(string, MAIL_REGEX)
    tokens.extend(_extract_by_rule(string, URL_REGEX))
    return tokens


def get_names(string: str) -> List[str]:
    return _extract_by_rule(string, NAME_REGEX)


def get_numeros(string: str) -> List[str]:
    return _extract_by_rule(string, NUMBER_REGEX)


def _extract_by_rule(string: str, rule: str) -> List[str]:
    tokens = []
    for match in re.finditer(rule, string):
        token = match.group(0)
        if TOKEN_MIN_LEN <= len(token) <= TOKEN_MAX_LEN:
            tokens.append(token)
    return tokens
