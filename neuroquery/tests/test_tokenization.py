import os
import tempfile

import pytest

import numpy as np
import pandas as pd

from neuroquery import tokenization

TEST_DATA = os.path.join(os.path.dirname(__file__), "data")
VOCABULARY_FILE = os.path.join(TEST_DATA, "vocabulary_list.csv")


def test_nltk_stop_words():
    stop_words = tokenization.nltk_stop_words()
    assert "the" in stop_words
    assert len(stop_words) > 100


def test_sklearn_stop_words():
    stop_words = tokenization.sklearn_stop_words()
    assert "the" in stop_words
    assert len(stop_words) > 100


def test_tokenizer():
    tokenizer = tokenization.Tokenizer()
    assert tokenizer("The m\u03b1chine-learning. algo 123xb;a\nbc") == [
        "The",
        "m\u03b1chine",
        "learning",
        "algo",
        "123xb",
        "bc",
    ]


def test_get_stemmer():
    sentence = "mice eating cheese".split()
    identity = tokenization._get_stemmer("identity")
    assert list(map(identity, sentence)) == sentence
    porter = tokenization._get_stemmer("porter_stemmer")
    assert list(map(porter, sentence)) == ["mice", "eat", "chees"]
    with pytest.raises(ValueError):
        tokenization._get_stemmer("other")
    try:
        wordnet = tokenization._get_stemmer("wordnet_lemmatizer")
        assert list(map(wordnet, sentence)) == ["mouse", "eating", "cheese"]
    except LookupError:
        # wordnet not installed
        return


def test_get_stop_words():
    stop = tokenization._get_stop_words("nltk")
    assert stop == tokenization.nltk_stop_words()
    stop = tokenization._get_stop_words("sklearn")
    assert stop == tokenization.sklearn_stop_words()
    stop = tokenization._get_stop_words(["sklearn", "nltk"])
    assert stop == {"sklearn", "nltk"}
    with pytest.raises(ValueError):
        tokenization._get_stop_words("other")


def test_standardizer():
    standardizer = tokenization.Standardizer("porter_stemmer")
    assert standardizer(["EatIng", "Cheese"]) == ["eat", "chees"]


def test_stop_word_filter():
    stop = tokenization.StopWordFilter(["the", "of"])
    assert isinstance(stop.stop_words_, set)
    assert stop(["and", "the", "they", "of"]) == ["and", "they"]
    standardizer = tokenization.Standardizer("porter_stemmer")
    stop = tokenization.StopWordFilter("nltk", standardizer)
    assert "yourselv" in stop.stop_words_
    assert stop(standardizer("do it yourselves computers".split())) == [
        "comput"
    ]


def test_build_phrase_map():
    phrases = [
        ("machine", "learning"),
        ("default", "mode", "network"),
        ("resting", "state"),
        ("learning",),
        ("network",),
        ("brain",),
        ("machine",),
        ("speech", "perception"),
        ("speech", "production"),
        ("speech",),
    ]
    phrase_map = tokenization._build_phrase_map(phrases)
    assert phrase_map == {
        "brain": {"": {}},
        "default": {"mode": {"network": {"": {}}}},
        "learning": {"": {}},
        "machine": {"": {}, "learning": {"": {}}},
        "network": {"": {}},
        "resting": {"state": {"": {}}},
        "speech": {"": {}, "perception": {"": {}}, "production": {"": {}}},
    }


def test_extract_phrases():
    phrases = [("machine",), ("machine", "learning"), ("algorithm",)]
    phrase_map = tokenization._build_phrase_map(phrases)
    sentence = "the new machine learning algorithm".split()
    all_phrases = tokenization._extract_phrases(phrase_map, sentence, "keep")
    assert all_phrases == [
        ("the",),
        ("new",),
        ("machine", "learning"),
        ("algorithm",),
    ]
    all_phrases = tokenization._extract_phrases(
        phrase_map, sentence, tokenization.OUT_OF_VOC_TOKEN
    )
    assert all_phrases == [
        (tokenization.OUT_OF_VOC_TOKEN,),
        (tokenization.OUT_OF_VOC_TOKEN,),
        ("machine", "learning"),
        ("algorithm",),
    ]
    all_phrases = tokenization._extract_phrases(phrase_map, sentence, "[]")
    assert all_phrases == [
        ("[the]",),
        ("[new]",),
        ("machine", "learning"),
        ("algorithm",),
    ]
    all_phrases = tokenization._extract_phrases(phrase_map, sentence, "{}")
    assert all_phrases == [
        ("{the}",),
        ("{new}",),
        ("machine", "learning"),
        ("algorithm",),
    ]


def test_load_vocabulary():
    voc = tokenization.load_vocabulary(VOCABULARY_FILE)
    assert len(voc) == 200
    assert ("working", "memory") in {v[0] for v in voc}
    with tempfile.TemporaryDirectory() as tmp_dir:
        data = pd.read_csv(VOCABULARY_FILE, header=None)
        voc_file = os.path.join(tmp_dir, "voc.csv")
        data.to_csv(voc_file, header=None, index=False)
        loaded = tokenization.load_vocabulary(voc_file)
        for ((w1, f1), (w2, f2)) in zip(voc, loaded):
            assert w1 == w2
            assert np.allclose(f1, f2)
        data.iloc[:, :1].to_csv(voc_file, header=None, index=False)
        loaded = tokenization.load_vocabulary(voc_file)
        assert len(loaded) == len(voc)
        for _, f in loaded:
            assert f == 1
        voc_file = os.path.join(tmp_dir, "voc.txt")
        with open(voc_file, "w") as f:
            for (w, _) in voc:
                f.write(" ".join(w))
                f.write("\n")
        loaded = tokenization.load_vocabulary(voc_file)
        assert len(loaded) == len(voc)
        for _, f in loaded:
            assert f == 1


def test_tuples_and_strings():
    tuples = [("a", "b"), ("c",), ("de", "fg")]
    strings = ["a b", "c", "de fg"]
    assert tokenization.tuple_sequence_to_strings(tuples) == strings
    assert tokenization.tuple_sequence_to_strings(strings) == strings
    assert tokenization.string_sequence_to_tuples(strings) == tuples
    assert tokenization.string_sequence_to_tuples(tuples) == tuples


@pytest.mark.parametrize("voc_mapping", ["auto", {}])
@pytest.mark.parametrize("with_frequencies", [True, False])
def test_tokenizing_pipeline(voc_mapping, with_frequencies):
    tok = tokenization.tokenizing_pipeline_from_vocabulary_file(
        VOCABULARY_FILE, voc_mapping=voc_mapping
    )
    if not with_frequencies:
        tok.frequencies = None
    if voc_mapping == {}:
        assert tok("the working memory group xyzzzz groups") == [
            "working memory",
            "group",
            "groups",
        ]
    else:
        assert tok("the working memory group xyzzzz groups") == [
            "working memory",
            "group",
            "group",
        ]
    assert tok.get_full_vocabulary(
        as_tuples=True
    ) == tokenization.string_sequence_to_tuples(tok.get_full_vocabulary())
    assert tok.get_vocabulary(
        as_tuples=True
    ) == tokenization.string_sequence_to_tuples(tok.get_vocabulary())
    if voc_mapping == "auto":
        assert len(tok.get_full_vocabulary()) == len(tok.get_vocabulary()) + 2
    else:
        assert len(tok.get_full_vocabulary()) == len(tok.get_vocabulary())
    assert len(tok.get_frequencies()) == len(tok.get_vocabulary())
    if with_frequencies:
        assert hasattr(tok, "frequencies_")
        assert len(tok.get_frequencies()) == len(tok.get_vocabulary())
    with tempfile.TemporaryDirectory() as tmp_dir:
        voc_file = os.path.join(tmp_dir, "voc_file.csv")
        tok.to_vocabulary_file(voc_file)
        loaded = tokenization.tokenizing_pipeline_from_vocabulary_file(
            voc_file, voc_mapping=voc_mapping
        )
        assert (
            loaded.vocabulary_mapping_.voc_mapping
            == tok.vocabulary_mapping_.voc_mapping
        )
        assert loaded.get_full_vocabulary() == tok.get_full_vocabulary()
        assert loaded.get_vocabulary() == tok.get_vocabulary()


def test_get_standardizing_inverse():
    std_inv = tokenization.get_standardizing_inverse(
        VOCABULARY_FILE,
        lambda t: tokenization.standardize_text(t, stemming="porter_stemmer"),
    )
    assert std_inv["memori"] == "memory"
    assert std_inv["work memori"] == "working memory"
    assert std_inv["nerv"] == "nerves"


def test_standardize_text():
    text = "One a the Word abcd-eft: --\nhello\t 1240"
    assert (
        tokenization.standardize_text(text) == "one word abcd eft hello 1240"
    )


def test_default_pipeline():
    text = "One a the Word abcd-eft: --\nhello\t 1240"
    pipe = tokenization.TokenizingPipeline(as_tuples=True)
    tok = pipe(text)
    assert tok == [
        ("one",),
        ("word",),
        ("abcd",),
        ("eft",),
        ("hello",),
        ("1240",),
    ]


def test_highlight_text():
    tok = tokenization.tokenizing_pipeline_from_vocabulary(
        [("one",), ("twenty", "three")]
    )
    tokens = tok("The One twenty plus TWENTY-three numbers", keep_pos=True)
    assert tokens == ["one", "twenty three"]
    highlighted = tokenization.etree.XML(
        tok.highlighted_text(
            extra_info=lambda p: {"is_large": p == "twenty three"}
        )
    )
    parts = highlighted.xpath("child::node()")
    assert len(parts) == 5
    assert parts[0] == "The "
    assert parts[1].get("standardized_form") == "one"
    assert parts[1].text == "One"
    assert parts[3].get("is_large") == "True"
    assert parts[1].get("is_large") == "False"
    printable = tokenization.get_printable_highlighted_text(
        tok.highlighted_text()
    )
    assert printable == (
        "The \x1b[94m[One]\x1b[0m twenty plus "
        "\x1b[94m[TWENTY-three]\x1b[0m numbers"
    )
    printable = tokenization.get_printable_highlighted_text(
        tok.highlighted_text(), replace=True
    )
    assert printable == (
        "The \x1b[92m[one]\x1b[0m twenty plus "
        "\x1b[92m[twenty three]\x1b[0m numbers"
    )
    html = tokenization.get_html_highlighted_text(tok.highlighted_text())
    assert html == (
        '<span>The <span style="background-color: LightBlue;">'
        'One</span> twenty plus <span style="background-color: '
        'LightBlue;">TWENTY-three</span> numbers</span>'
    )
    tok.print_highlighted_text()
    tokens = tok(".+ --", keep_pos=True)
    assert tokens == []
    highlighted = tokenization.etree.XML(tok.highlighted_text())
    parts = highlighted.xpath("child::node()")
    assert len(parts) == 1
    assert parts[0] == ".+ --"
    html = tokenization.get_html_highlighted_text(
        tok.highlighted_text(), standalone=True
    )
    assert html == (
        '<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.0 Transitional'
        '//EN" "http://www.w3.org/TR/REC-html40/loose.dtd">\n'
        '<html><head><meta charset="UTF-8"><title>highlighted '
        "text</title></head><body><span>.+ --</span></body></html>"
    )


def test_make_voc_mapping():
    voc = [
        ("experiment",),
        ("experiments",),
        ("experience"),
        ("experimentss",),
    ]
    freq = [1.0, 0.5, 0.2, 0.01]
    voc_mapping = tokenization.make_vocabulary_mapping(voc, freq)
    assert voc_mapping == {
        ("experiments",): ("experiment",),
        ("experimentss",): ("experiment",),
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        df = pd.DataFrame(
            {"term": tokenization.tuple_sequence_to_strings(voc), "freq": freq}
        )
        voc_file = os.path.join(tmp_dir, "voc.csv")
        df.to_csv(voc_file, header=None, index=False)
        # voc_mapping = tokenization.load_voc_mapping(voc_file)
        # assert voc_mapping == {('experiments',): ('experiment',)}
        pipe = tokenization.tokenizing_pipeline_from_vocabulary_file(
            voc_file, voc_mapping="auto"
        )
        assert pipe.vocabulary_mapping_.voc_mapping == {
            ("experiments",): ("experiment",),
            ("experimentss",): ("experiment",),
        }
        pipe = tokenization.tokenizing_pipeline_from_vocabulary(
            voc, voc_mapping="auto", frequencies=freq
        )
        assert pipe.vocabulary_mapping_.voc_mapping == {
            ("experiments",): ("experiment",),
            ("experimentss",): ("experiment",),
        }


def test_unigrams():
    voc = ["working memory", "brain", "memory"]
    op = tokenization.unigram_operator(voc).A
    assert np.allclose(op, [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    freq = tokenization.add_unigram_frequencies([1, 1, 1], voc=voc)
    assert np.allclose(freq, [1, 1, 2])


def test_text_vectorizer():
    docs = ["attention Encoding-language", "routine fixation", "ab and action"]
    vect = tokenization.TextVectorizer.from_vocabulary_file(VOCABULARY_FILE)
    transformed = vect(docs)
    assert np.allclose(transformed.A[2, :4], [0.83618742, 0.5484438, 0.0, 0.0])
    vect = tokenization.TextVectorizer.from_vocabulary_file(
        VOCABULARY_FILE, use_idf=False, norm="l1", add_unigrams=False
    )
    transformed = vect(docs)
    assert np.allclose(transformed.A[2, :4], [0.5, 0.5, 0.0, 0.0])
    vect = tokenization.TextVectorizer.from_vocabulary(
        vect.get_vocabulary(), norm="l1"
    )
    transformed = vect(docs)
    assert np.allclose(transformed.A[2, :4], [0.5, 0.5, 0.0, 0.0])
    assert vect.get_feature_names() == vect.tokenizer.get_vocabulary()
    assert vect.get_full_vocabulary() == vect.tokenizer.get_full_vocabulary()
