import os

import pytest

from neuroquery import tokenization

TEST_DATA = os.path.join(os.path.dirname(__file__), 'data')
VOCABULARY_FILE = os.path.join(TEST_DATA, 'vocabulary_list.csv')


def test_nltk_stop_words():
    stop_words = tokenization.nltk_stop_words()
    assert 'the' in stop_words
    assert len(stop_words) > 100


def test_sklearn_stop_words():
    stop_words = tokenization.sklearn_stop_words()
    assert 'the' in stop_words
    assert len(stop_words) > 100


def test_tokenizer():
    tokenizer = tokenization.Tokenizer()
    assert (
        tokenizer('The m\u03b1chine-learning. algo 123xb;a\nbc')
        == ['The', 'm\u03b1chine', 'learning', 'algo', '123xb', 'bc'])


def test_get_stemmer():
    sentence = 'mice eating cheese'.split()
    identity = tokenization._get_stemmer('identity')
    assert list(map(identity, sentence)) == sentence
    porter = tokenization._get_stemmer('porter_stemmer')
    assert list(map(porter, sentence)) == ['mice', 'eat', 'chees']
    with pytest.raises(ValueError):
        tokenization._get_stemmer('other')
    try:
        wordnet = tokenization._get_stemmer('wordnet_lemmatizer')
        assert (
            list(map(wordnet, sentence)) == ['mouse', 'eating', 'cheese'])
    except LookupError:
        # wordnet not installed
        return


def test_get_stop_words():
    stop = tokenization._get_stop_words('nltk')
    assert stop == tokenization.nltk_stop_words()
    stop = tokenization._get_stop_words('sklearn')
    assert stop == tokenization.sklearn_stop_words()
    stop = tokenization._get_stop_words(['sklearn', 'nltk'])
    assert stop == {'sklearn', 'nltk'}
    with pytest.raises(ValueError):
        tokenization._get_stop_words('other')


def test_standardizer():
    standardizer = tokenization.Standardizer('porter_stemmer')
    assert standardizer(['EatIng', 'Cheese']) == ['eat', 'chees']


def test_stop_word_filter():
    stop = tokenization.StopWordFilter(['the', 'of'])
    assert isinstance(stop.stop_words_, set)
    assert stop(['and', 'the', 'they', 'of']) == ['and', 'they']
    standardizer = tokenization.Standardizer('porter_stemmer')
    stop = tokenization.StopWordFilter('nltk', standardizer)
    assert 'yourselv' in stop.stop_words_
    assert (
        stop(standardizer('do it yourselves computers'.split())) == ['comput'])


def test_build_phrase_map():
    phrases = [('machine', 'learning'), ('default', 'mode',
                                         'network'), ('resting', 'state'),
               ('learning', ), ('network', ), ('brain', ), ('machine', ),
               ('speech', 'perception'), ('speech',
                                          'production'), ('speech', )]
    phrase_map = tokenization._build_phrase_map(phrases)
    assert (
        phrase_map == {
            'brain': {
                '': {}
            },
            'default': {
                'mode': {
                    'network': {
                        '': {}
                    }
                }
            },
            'learning': {
                '': {}
            },
            'machine': {
                '': {},
                'learning': {
                    '': {}
                }
            },
            'network': {
                '': {}
            },
            'resting': {
                'state': {
                    '': {}
                }
            },
            'speech': {
                '': {},
                'perception': {
                    '': {}
                },
                'production': {
                    '': {}
                }
            }
        })


def test_extract_phrases():
    phrases = [('machine', ), ('machine', 'learning'), ('algorithm', )]
    phrase_map = tokenization._build_phrase_map(phrases)
    sentence = 'the new machine learning algorithm'.split()
    all_phrases = tokenization._extract_phrases(phrase_map, sentence, 'keep')
    assert all_phrases == [('the', ), ('new', ),
                           ('machine', 'learning'), ('algorithm', )]
    all_phrases = tokenization._extract_phrases(phrase_map, sentence,
                                                tokenization.OUT_OF_VOC_TOKEN)
    assert (all_phrases == [(tokenization.OUT_OF_VOC_TOKEN, ),
                            (tokenization.OUT_OF_VOC_TOKEN, ),
                            ('machine', 'learning'), ('algorithm', )])
    known_phrases = tokenization._extract_phrases(phrase_map, sentence,
                                                  'ignore')
    assert known_phrases == [('machine', 'learning'), ('algorithm', )]


def test_load_vocabulary():
    voc = tokenization.load_vocabulary(VOCABULARY_FILE)
    assert len(voc) == 200
    assert ('working', 'memory') in {v[0] for v in voc}


def test_tuples_and_strings():
    tuples = [('a', 'b'), ('c', ), ('de', 'fg')]
    strings = ['a b', 'c', 'de fg']
    assert tokenization.tuple_sequence_to_strings(tuples) == strings
    assert tokenization.tuple_sequence_to_strings(strings) == strings
    assert tokenization.string_sequence_to_tuples(strings) == tuples
    assert tokenization.string_sequence_to_tuples(tuples) == tuples


def test_tokenizing_pipeline():
    tok = tokenization.tokenizing_pipeline_from_vocabulary_file(
        VOCABULARY_FILE)
    assert tok('the working memory group xyzzzz') == [
        'working memory', 'group'
    ]


def test_get_standardizing_inverse():
    std_inv = tokenization.get_standardizing_inverse(
        VOCABULARY_FILE,
        lambda t: tokenization.standardize_text(t, stemming='porter_stemmer'))
    assert std_inv['memori'] == 'memory'
    assert std_inv['work memori'] == 'working memory'
    assert std_inv['nerv'] == 'nerves'


def test_highlight_text():
    tok = tokenization.tokenizing_pipeline_from_vocabulary(
        [('one',), ('twenty', 'three')])
    tokens = tok('The One twenty plus TWENTY-three numbers', keep_pos=True)
    assert tokens == ['one', 'twenty three']
    highlighted = tokenization.etree.XML(tok.highlighted_text())
    parts = highlighted.xpath('child::node()')
    assert len(parts) == 5
    assert parts[0] == 'The '
    assert parts[1].get('standardized_form') == 'one'
    assert parts[1].text == 'One'
