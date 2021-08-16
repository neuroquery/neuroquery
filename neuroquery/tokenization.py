import json
import pathlib
from xml.sax.saxutils import escape

from scipy import sparse
import regex
import sklearn
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from lxml import etree


WORD_PATTERN = r"(?:(?:\p{L}|\p{N}){2,}|[IVXivx0-9])"

OUT_OF_VOC_TOKEN = (
    "https://gitlab.inria.fr/parietal/jerome_dockes/tree/"
    "master/semantic_structure/corpus_manip/xml/ns#"
    "out-of-vocabulary-token"
)

RARE_TOKEN = (
    "https://gitlab.inria.fr/parietal/jerome_dockes/tree/"
    "master/semantic_structure/corpus_manip/xml/ns#"
    "rare-token"
)

_TERM_BLUE = "\033[94m"
_TERM_GREEN = "\033[92m"
_TERM_ENDC = "\033[0m"

# not synonyms despite high jaro-winkler similarity
_DIFFERENT_WORDS = {
    ("addiction", "addition"),
    ("imitation", "limitation"),
    ("preference", "reference"),
    ("asymmetric", "symmetric"),
    ("mediation", "medication"),
    ("mediation", "meditation"),
    ("preferential", "referential"),
    ("asymptomatic", "symptomatic"),
    ("covert attention", "overt attention"),
}


def _identity(arg):
    return arg


def nltk_stop_words():
    with open(
        str(pathlib.Path(__file__).parent / "data" / "nltk_stop_words.txt")
    ) as f:
        words = {line.strip() for line in f}.difference({""})
    return words


def sklearn_stop_words():
    return set(sklearn.feature_extraction.text.ENGLISH_STOP_WORDS)


class Tokenizer(object):
    def __init__(self, token_pattern=WORD_PATTERN):
        self.token_pattern_ = token_pattern
        self._compiled_regexp = regex.compile(token_pattern)
        self.match_positions = None

    def __call__(self, text, keep_pos=False):
        if keep_pos:
            return self._tokenize_keep_pos(text)
        return self._tokenize(text)

    def _tokenize(self, text):
        return self._compiled_regexp.findall(str(text))

    def _tokenize_keep_pos(self, text):
        matches = self._compiled_regexp.finditer(str(text))
        result = []
        positions = []
        for m in matches:
            result.append(m.group())
            positions.append(m.span())
        self.match_positions = np.asarray(positions)
        return result


def _get_stemmer(stemming_kind):
    if stemming_kind == "identity":
        return _identity
    if stemming_kind == "porter_stemmer":
        try:
            import nltk
        except ImportError:
            raise ImportError(
                "nltk must be installed to use the Porter stemmer")
        return nltk.PorterStemmer().stem
    if stemming_kind == "wordnet_lemmatizer":
        try:
            import nltk
        except ImportError:
            raise ImportError(
                "nltk must be installed to use the WordNet lemmatizer")
        return nltk.stem.WordNetLemmatizer().lemmatize
    raise ValueError('invalid value for "stemming_kind".')


def _get_stop_words(stop_words):
    if not isinstance(stop_words, str):
        return set(stop_words)
    if stop_words == "nltk":
        return nltk_stop_words()
    if stop_words == "sklearn":
        return sklearn_stop_words()
    raise ValueError(stop_words)


class Standardizer(object):
    def __init__(self, stemming="identity"):
        self.stemming_ = stemming
        self._stemmer = _get_stemmer(stemming)

    def _standardize_token(self, token):
        return self._stemmer(str.lower(token))

    def __call__(self, tokens):
        return list(map(self._standardize_token, tokens))


class StopWordFilter(object):
    def __init__(self, stop_words, standardizer=Standardizer()):
        stop_words = _get_stop_words(stop_words)
        self.stop_words_ = set(standardizer(stop_words))
        self.kept_tokens = None

    def __call__(self, tokens, keep_pos=False):
        if keep_pos:
            return self._filter_stop_words_keep_pos(tokens)
        return self._filter_stop_words(tokens)

    def _filter_stop_words(self, tokens):
        return [t for t in tokens if t not in self.stop_words_]

    def _filter_stop_words_keep_pos(self, tokens):
        result = []
        kept = []
        for i, tok in enumerate(tokens):
            if tok not in self.stop_words_:
                kept.append(i)
                result.append(tok)
        self.kept_tokens = np.asarray(kept)
        return result


def _build_phrase_map(phrases):
    phrase_map = {}
    for phrase in phrases:
        _update_phrase_map(phrase_map, phrase)
    return phrase_map


def _update_phrase_map(phrase_map, phrase):
    head = phrase[0]
    tail = phrase[1:]
    head_map = phrase_map.get(head, {})
    if not len(tail):
        head_map[""] = {}
        phrase_map[head] = head_map
        return
    _update_phrase_map(head_map, tail)
    phrase_map[head] = head_map
    return


def _transform_out_of_voc_token(token, out_of_voc):
    if out_of_voc == "ignore":
        return tuple()
    if out_of_voc == "keep":
        return (token,)
    if out_of_voc == "[]":
        return ("[{}]".format(token),)
    if out_of_voc == "{}":
        return ("{{{}}}".format(token),)
    return (out_of_voc,)


def _extract_next_phrase(phrase_map, sentence, out_of_voc):
    if not sentence:
        return tuple(), sentence
    sentence_head = sentence[0]
    sentence_tail = sentence[1:]
    if sentence_head not in phrase_map:
        return (
            _transform_out_of_voc_token(sentence_head, out_of_voc),
            sentence_tail,
        )
    phrase_head = sentence_head
    phrase_tail, new_sentence_tail = _extract_next_phrase(
        phrase_map[phrase_head], sentence_tail, out_of_voc="ignore"
    )
    if phrase_tail:
        return (phrase_head, *phrase_tail), new_sentence_tail
    if "" in phrase_map[phrase_head]:
        return (phrase_head,), sentence_tail
    return (
        _transform_out_of_voc_token(sentence_head, out_of_voc),
        sentence_tail,
    )


def _extract_phrases(phrase_map, sentence, out_of_voc):
    phrases = []
    while sentence:
        next_phrase, sentence = _extract_next_phrase(
            phrase_map, sentence, out_of_voc
        )
        if next_phrase:
            phrases.append(next_phrase)
    return phrases


class PhraseExtractor(object):
    def __init__(self, phrases, out_of_voc="ignore"):
        self.phrases_ = set(phrases)
        self.out_of_voc_ = out_of_voc
        self._phrase_map = _build_phrase_map(phrases)
        self.phrase_positions = None

    def __call__(self, sentence, keep_pos=False):
        if keep_pos:
            return self._extract_phrases_keep_pos(sentence)
        return _extract_phrases(self._phrase_map, sentence, self.out_of_voc_)

    def _extract_phrases_keep_pos(self, sentence):
        out_of_voc = {"ignore": OUT_OF_VOC_TOKEN}.get(
            self.out_of_voc_, self.out_of_voc_
        )
        extracted = _extract_phrases(self._phrase_map, sentence, out_of_voc)
        result = []
        i = 0
        positions = []
        for phrase in extracted:
            if (
                phrase != (OUT_OF_VOC_TOKEN,)
                or self.out_of_voc_ == OUT_OF_VOC_TOKEN
            ):
                result.append(phrase)
                positions.append((i, i + len(phrase)))
            i += len(phrase)
        self.phrase_positions = np.asarray(positions)
        return result


class VocabularyMapping(object):
    def __init__(self, voc_mapping={}):
        self.voc_mapping = voc_mapping

    def __call__(self, phrases):
        if not self.voc_mapping:
            return phrases
        return [self.voc_mapping.get(p, p) for p in phrases]


class TokenizingPipeline(object):
    def __init__(
        self,
        vocabulary_mapping=None,
        phrase_extractor=None,
        stop_word_filter=None,
        standardizer=None,
        tokenizer=None,
        as_tuples=False,
        keep_pos=False,
        frequencies=None,
        raw_vocabulary=None,
    ):
        self.vocabulary_mapping_ = vocabulary_mapping
        self.phrase_extractor_ = phrase_extractor
        self.stop_word_filter_ = stop_word_filter
        self.standardizer_ = standardizer
        self.tokenizer_ = tokenizer
        self.as_tuples_ = as_tuples
        self.keep_pos = keep_pos
        self.frequencies = frequencies
        if self.frequencies is not None:
            self.frequencies.index = tuple_sequence_to_strings(
                self.frequencies.index
            )
        if raw_vocabulary is not None:
            self.raw_vocabulary = tuple_sequence_to_strings(raw_vocabulary)
        else:
            self.raw_vocabulary = None
        if self.tokenizer_ is None:
            self.tokenizer_ = Tokenizer()
        if self.standardizer_ is None:
            self.standardizer_ = Standardizer()
        if self.stop_word_filter_ is None:
            self.stop_word_filter_ = StopWordFilter("nltk", self.standardizer_)
        if self.phrase_extractor_ is None:
            self.phrase_extractor_ = PhraseExtractor([], out_of_voc="keep")
        if self.vocabulary_mapping_ is None:
            self.vocabulary_mapping_ = VocabularyMapping()
        self.positions = None
        self.extracted_phrases = None
        self.raw_text = None

    def to_vocabulary_file(self, voc_file):
        if self.frequencies is not None:
            freq = self.frequencies
        else:
            freq = np.ones(len(self.raw_vocabulary))
        pd.Series(freq, index=self.raw_vocabulary).to_csv(
            voc_file, header=False
        )
        _save_voc_mapping(
            voc_file,
            self.vocabulary_mapping_.voc_mapping,
            stemming=self.standardizer_.stemming_,
        )

    def __call__(self, text, keep_pos=None):
        keep_pos = self.keep_pos if keep_pos is None else keep_pos
        tokenized = self.vocabulary_mapping_(
            self.phrase_extractor_(
                self.stop_word_filter_(
                    self.standardizer_(
                        self.tokenizer_(text, keep_pos=keep_pos)
                    ),
                    keep_pos=keep_pos,
                ),
                keep_pos=keep_pos,
            )
        )
        if self.as_tuples_:
            phrases = tokenized
        else:
            phrases = tuple_sequence_to_strings(tokenized)
        if keep_pos:
            self._store_positions(phrases, text)
        return phrases

    def _store_positions(self, phrases, text):
        self.raw_text = text
        self.extracted_phrases = phrases
        if not phrases:
            self.positions = np.asarray([])
            return
        token_positions = self.tokenizer_.match_positions[
            self.stop_word_filter_.kept_tokens
        ]
        start_positions = token_positions[
            self.phrase_extractor_.phrase_positions[:, 0]
        ][:, 0]
        end_positions = token_positions[
            self.phrase_extractor_.phrase_positions[:, 1] - 1
        ][:, 1]
        self.positions = np.asarray([start_positions, end_positions]).T

    def get_full_vocabulary(self, as_tuples=False):
        voc = self.phrase_extractor_.phrases_
        if as_tuples:
            return sorted(voc)
        return sorted(tuple_sequence_to_strings(voc))

    def get_vocabulary(self, as_tuples=False):
        voc = self.phrase_extractor_.phrases_.difference(
            self.vocabulary_mapping_.voc_mapping.keys()
        )
        if as_tuples:
            return sorted(voc)
        return sorted(tuple_sequence_to_strings(voc))

    def get_frequencies(self):
        if hasattr(self, "frequencies_"):
            return self.frequencies_
        if self.frequencies is None:
            return np.ones(len(self.get_vocabulary()))
        idx = tuple_sequence_to_strings(
            [(self(w) or [""])[0] for w in self.frequencies.index]
        )
        freq = self.frequencies.groupby(idx).sum()
        self.frequencies_ = freq.loc[self.get_vocabulary()].values
        return self.frequencies_

    def highlighted_text(self, extra_info=None):
        return highlight_text(
            self.raw_text,
            self.extracted_phrases,
            self.positions,
            extra_info=extra_info,
        )

    def print_highlighted_text(self, replace=False):
        print_highlighted_text(self.highlighted_text(), replace=replace)


def highlight_text(text, phrases, positions, extra_info=None):
    phrases = tuple_sequence_to_strings(phrases)
    snippets = []
    prev = 0
    snippets.append("<highlighted_text>")
    for phrase, (start, stop) in zip(phrases, positions):
        snippets.append(escape(text[prev:start]))
        attributes = {"standardized_form": phrase}
        if extra_info is not None:
            attributes.update(extra_info(phrase))
        attr_str = " ".join(
            ['{}="{}"'.format(k, v) for (k, v) in attributes.items()]
        )
        snippets.append("<extracted_phrase {}>".format(attr_str))
        snippets.append(escape(text[start:stop]))
        snippets.append("</extracted_phrase>")
        prev = stop
    snippets.append(escape(text[prev:]))
    snippets.append("</highlighted_text>")
    return "".join(snippets)


def get_printable_highlighted_text(text, replace=False):
    highlighted = etree.XML(text)
    parts = []
    for node in highlighted.xpath("child::node()"):
        if isinstance(node, etree._Element):
            if replace:
                parts.extend(
                    [
                        _TERM_GREEN,
                        "[",
                        node.get("standardized_form"),
                        "]",
                        _TERM_ENDC,
                    ]
                )
            else:
                parts.extend([_TERM_BLUE, "[", node.text, "]", _TERM_ENDC])
        else:
            parts.append(str(node))
    return "".join(parts)


def print_highlighted_text(text, replace=False):
    print(get_printable_highlighted_text(text, replace=replace))


def get_html_highlighted_text(text, standalone=False):
    stylesheet_path = pathlib.Path(__file__).parent / "data" / "highlight.xsl"
    stylesheet = etree.XSLT(etree.parse(str(stylesheet_path)))
    html = stylesheet(etree.XML(text, parser=etree.XMLParser(recover=True)))
    if not standalone:
        html = html.findall("body")[0][0]
    return etree.tostring(html, method="html", encoding="unicode")


def load_vocabulary(vocabulary_file, token_pattern=WORD_PATTERN):
    tokenizer = Tokenizer(token_pattern=token_pattern)
    if vocabulary_file.endswith(".csv"):
        try:
            word_freq = pd.read_csv(
                vocabulary_file,
                header=None,
                encoding="utf-8",
                usecols=[0, 1],
                na_values=[],
                keep_default_na=False,
            ).values
            words, frequencies = word_freq.T
        except ValueError:
            words = pd.read_csv(
                vocabulary_file,
                header=None,
                encoding="utf-8",
                usecols=[0],
                na_values=[],
                keep_default_na=False,
            ).values.ravel()
            frequencies = np.ones(words.shape)
    else:
        with open(vocabulary_file, "r", encoding="utf-8") as fh:
            words = fh.readlines()
            frequencies = np.ones(len(words), dtype=int)
    vocabulary = []
    for word, freq in zip(words, frequencies):
        tokens = tuple(tokenizer(word))
        if tokens:
            vocabulary.append((tokens, freq))
    return vocabulary


def tokenizing_pipeline_from_vocabulary(
    vocabulary,
    frequencies=None,
    stemming="identity",
    stop_words="nltk",
    out_of_voc="ignore",
    voc_mapping={},
    token_pattern=WORD_PATTERN,
    as_tuples=False,
):
    if frequencies is None:
        frequencies = np.ones(len(vocabulary))
    tokenizer = Tokenizer(token_pattern)
    standardizer = Standardizer(stemming=stemming)
    stop_word_filter = StopWordFilter(stop_words, standardizer)
    std = TokenizingPipeline(
        phrase_extractor=PhraseExtractor(phrases=[], out_of_voc="keep"),
        stop_word_filter=stop_word_filter,
        standardizer=standardizer,
        tokenizer=tokenizer,
    )
    phrases = {tuple(std(phrase)) for phrase in vocabulary}.difference({()})
    phrase_extractor = PhraseExtractor(phrases, out_of_voc=out_of_voc)
    vocabulary = string_sequence_to_tuples(vocabulary)
    if voc_mapping == "auto":
        voc_mapping = make_vocabulary_mapping(
            vocabulary, frequencies, stemming=stemming
        )
    voc_mapper = VocabularyMapping(voc_mapping)
    return TokenizingPipeline(
        voc_mapper,
        phrase_extractor,
        stop_word_filter,
        standardizer,
        tokenizer,
        as_tuples=as_tuples,
        raw_vocabulary=vocabulary,
        frequencies=pd.Series(frequencies, index=vocabulary),
    )


def tokenizing_pipeline_from_vocabulary_file(
    vocabulary_file,
    voc_mapping={},
    stemming="identity",
    stop_words="nltk",
    out_of_voc="ignore",
    token_pattern=WORD_PATTERN,
    as_tuples=False,
):
    phrases = load_vocabulary(vocabulary_file, token_pattern=token_pattern)
    freq = [p[1] for p in phrases]
    phrases = [p[0] for p in phrases]
    if voc_mapping == "auto":
        voc_mapping = load_voc_mapping(
            vocabulary_file, stemming=stemming, token_pattern=token_pattern
        )
    return tokenizing_pipeline_from_vocabulary(
        phrases,
        frequencies=freq,
        stemming=stemming,
        stop_words=stop_words,
        out_of_voc=out_of_voc,
        token_pattern=token_pattern,
        voc_mapping=voc_mapping,
        as_tuples=as_tuples,
    )


def _save_voc_mapping(vocabulary_file, voc_mapping, stemming="identity"):
    voc_mapping_file = pathlib.Path(
        "{}_voc_mapping_{}.json".format(vocabulary_file, stemming)
    )
    with open(str(voc_mapping_file), "w") as f:
        mapping = dict(
            zip(
                tuple_sequence_to_strings(voc_mapping.keys()),
                tuple_sequence_to_strings(voc_mapping.values()),
            )
        )
        json.dump(mapping, f)
    return str(voc_mapping_file)


def load_voc_mapping(
    vocabulary_file, stemming="identity", token_pattern=WORD_PATTERN
):
    voc_mapping_file = pathlib.Path(
        "{}_voc_mapping_{}.json".format(vocabulary_file, stemming)
    )
    if voc_mapping_file.is_file():
        with open(str(voc_mapping_file)) as f:
            mapping = json.load(f)
        return dict(
            zip(
                string_sequence_to_tuples(mapping.keys()),
                string_sequence_to_tuples(mapping.values()),
            )
        )
    phrases = load_vocabulary(vocabulary_file, token_pattern=token_pattern)
    freq = [p[1] for p in phrases]
    phrases = [p[0] for p in phrases]
    voc_mapping = make_vocabulary_mapping(phrases, freq, stemming=stemming)
    voc_mapping_file = _save_voc_mapping(
        vocabulary_file, voc_mapping, stemming=stemming
    )
    print("voc mapping saved in {}".format(voc_mapping_file))
    return voc_mapping


def tuple_to_string(phrase, delimiter="\u0020"):
    if isinstance(phrase, tuple):
        return delimiter.join(phrase)
    return phrase


def tuple_sequence_to_strings(tuples, delimiter="\u0020"):
    return list(map(lambda t: tuple_to_string(t, delimiter), tuples))


def string_to_tuple(phrase, delimiter="\u0020"):
    if isinstance(phrase, str):
        return tuple(phrase.split(delimiter))
    return phrase


def string_sequence_to_tuples(strings, delimiter="\u0020"):
    return list(map(lambda s: string_to_tuple(s, delimiter), strings))


def standardize_text(text, stop_words="nltk", stemming="identity"):
    pipeline = TokenizingPipeline(
        VocabularyMapping({}),
        PhraseExtractor(phrases=[], out_of_voc="keep"),
        StopWordFilter(stop_words),
        Standardizer(stemming=stemming),
        Tokenizer(),
    )
    return " ".join(pipeline(text))


def get_standardizing_inverse(vocabulary_file, standardizer):
    vocabulary = load_vocabulary(vocabulary_file)
    vocabulary = pd.DataFrame(
        vocabulary, columns=["original_phrase", "n_occurrences"]
    )
    vocabulary["standardized_phrase"] = [
        " ".join([standardizer(tok) for tok in phrase])
        for phrase in vocabulary["original_phrase"]
    ]
    vocabulary.sort_values("n_occurrences", ascending=False, inplace=True)
    vocabulary.drop_duplicates("standardized_phrase", inplace=True)
    vocabulary.set_index("standardized_phrase", inplace=True)
    vocabulary["original_phrase"] = tuple_sequence_to_strings(
        vocabulary["original_phrase"]
    )
    voc_map = vocabulary["original_phrase"].to_dict()
    return voc_map


def unigram_operator(vocabulary):
    word_to_idx = pd.Series(np.arange(len(vocabulary)), index=vocabulary)
    row_idx, col_idx, data = [], [], []
    for word in word_to_idx.index:
        row_idx.append(word_to_idx[word])
        col_idx.append(word_to_idx[word])
        data.append(1)
        if " " in word:
            for unigram in word.split():
                if unigram in word_to_idx.index:
                    row_idx.append(word_to_idx[word])
                    col_idx.append(word_to_idx[unigram])
                    data.append(1)
    data = np.asarray(data, dtype="float32")
    return sparse.csr_matrix(
        (data, (row_idx, col_idx)), shape=(len(word_to_idx), len(word_to_idx))
    )


def add_unigram_frequencies(frequencies, unigram_op=None, voc=None):
    if not sparse.issparse(frequencies):
        frequencies = np.asarray(frequencies)
    if unigram_op is None:
        unigram_op = unigram_operator(voc)
    return safe_sparse_dot(frequencies, unigram_op)


def _jaro_w(a, b, jw):
    a_s, b_s = a.split(), b.split()
    if len(a_s) != len(b_s):
        return jw(a, b, 1 / 50.0)
    return min(jw(a_i, b_i, 1 / 10.0) for (a_i, b_i) in zip(a_s, b_s))


def _similar_words(voc):
    from Levenshtein import jaro_winkler as jw

    s = [_jaro_w(a, b, jw) for a in voc for b in voc]
    ss = np.reshape(s, (len(voc), len(voc)))
    del s
    pairs = np.nonzero(ss >= 0.96)
    pairs = [(voc[a], voc[b]) for (a, b) in zip(*pairs)]
    pairs = {
        (a, b)
        for (a, b) in pairs
        if (a < b)
        and (not regex.match(r".*(\d|\b[ivx]+\b).*", a))
        and (not regex.match(r".*(\d|\b[ivx]+\b).*", b))
    }
    return pairs.difference(_DIFFERENT_WORDS)


def _choose_pairs(pairs, freq):
    # TODO: fix
    pairs = sorted(pairs, key=lambda p: -max(freq[p[0]], freq[p[1]]))
    mapping = {}
    for a, b in pairs:
        if freq[a] > freq[b]:
            a, b = b, a
        if b == a + "s":
            a, b = b, a
        if b in mapping:
            mapping[a] = mapping[b]
        else:
            mapping[a] = b
    return mapping


def make_vocabulary_mapping(phrases, frequencies, stemming="identity"):
    standardizer = Standardizer(stemming=stemming)
    if frequencies is None:
        frequencies = np.ones(len(phrases))
    freq = pd.DataFrame(frequencies, columns=["freq"])
    freq["std_phrase"] = [" ".join(standardizer(p)) for p in phrases]
    freq = freq.groupby("std_phrase").sum()["freq"]
    pairs = _similar_words(freq.index)
    mapping = _choose_pairs(pairs, freq)
    return {string_to_tuple(k): string_to_tuple(v) for k, v in mapping.items()}


class TextVectorizer(object):
    @classmethod
    def from_vocabulary_file(
        cls,
        voc_file,
        voc_mapping="auto",
        stemming="identity",
        stop_words="nltk",
        out_of_voc="ignore",
        token_pattern=WORD_PATTERN,
        use_idf=True,
        norm="l2",
        add_unigrams=True,
    ):
        tokenizer = tokenizing_pipeline_from_vocabulary_file(
            voc_file,
            voc_mapping=voc_mapping,
            stemming=stemming,
            stop_words=stop_words,
            out_of_voc=out_of_voc,
            token_pattern=token_pattern,
        )
        return cls(
            tokenizer, use_idf=use_idf, norm=norm, add_unigrams=add_unigrams
        )

    @classmethod
    def from_vocabulary(
        cls,
        voc,
        frequencies=None,
        voc_mapping={},
        stemming="identity",
        stop_words="nltk",
        out_of_voc="ignore",
        token_pattern=WORD_PATTERN,
        use_idf=True,
        norm="l2",
        add_unigrams=True,
    ):
        tokenizer = tokenizing_pipeline_from_vocabulary(
            voc,
            frequencies=frequencies,
            voc_mapping=voc_mapping,
            stemming=stemming,
            stop_words=stop_words,
            out_of_voc=out_of_voc,
            token_pattern=token_pattern,
        )
        return cls(
            tokenizer, use_idf=use_idf, norm=norm, add_unigrams=add_unigrams
        )

    def __init__(self, tokenizer, use_idf=True, norm="l2", add_unigrams=True):
        self.tokenizer = tokenizer
        self.use_idf = use_idf
        self.norm = norm
        self.add_unigrams = add_unigrams

    def to_vocabulary_file(self, voc_file):
        return self.tokenizer.to_vocabulary_file(voc_file)

    def fit(self, *args, **kwargs):
        self.counter_ = CountVectorizer(
            analyzer=self.tokenizer,
            vocabulary=self.tokenizer.get_vocabulary(),
            min_df=0,
        ).fit([])
        if self.use_idf:
            self.idf_ = -np.log(self.tokenizer.get_frequencies()) + 1
            self._idf_diag = sparse.spdiags(
                self.idf_,
                diags=0,
                m=self.idf_.shape[0],
                n=self.idf_.shape[0],
                format="csr",
            )
        else:
            self._idf_diag = sparse.eye(
                len(self.tokenizer.get_vocabulary()),
                format="csr",
                dtype="float32",
            )
        if self.add_unigrams:
            self.unigram_op_ = unigram_operator(self.get_vocabulary())
        else:
            self.unigram_op_ = sparse.eye(
                len(self.tokenizer.get_vocabulary()),
                format="csr",
                dtype="float32",
            )
        return self

    def transform(self, docs):
        if not hasattr(self, "counter_"):
            self.fit()
        counts = self.counter_.transform(docs)
        counts = counts.dot(self.unigram_op_)
        counts = counts.dot(self._idf_diag)
        if self.norm is not None:
            counts = normalize(counts, norm=self.norm, axis=1, copy=False)
        return counts

    def __call__(self, docs):
        return self.transform(docs)

    def get_vocabulary(self):
        return self.tokenizer.get_vocabulary()

    def get_feature_names(self):
        return self.tokenizer.get_vocabulary()

    def get_full_vocabulary(self):
        return self.tokenizer.get_full_vocabulary()
