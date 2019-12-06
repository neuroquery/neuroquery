"""Functions to manipulate input coordinates."""


def build_index(I):
    """Build decode & encode dictionnaries of the given iterable.

    Parameters
    ----------
    I : iterable or string
        Iterable from which to build the dictionaries. If path given, open
        the corresponding file.

    Returns
    -------
    encode : dict
        key : index of the object found in the iterable.
        value : object at the specified index.
    decode : dict
        Reverse of encode.
        key : object found in the iterable.
        value : index of the object found in the iterable.

    """
    if isinstance(I, str):
        I = open(I)

    decode = dict(enumerate(line.strip() for line in I))
    encode = {v: k for k, v in decode.items()}

    return encode, decode


def select_pmids(coordinates, pmids):
    """Keep only the given pmids in the given coordinates data frame.

    Parameters
    ----------
    coordinates : pandas.DataFrame
        Data frame storing the peak coordinates.
    pmids : list of int
        Store the publication ids to keep in the data frame.

    Returns
    -------
    pandas.DataFrame
        Same columns as the input data frame but with specified pmids (rows)
        eliminated.

    """
    return coordinates[coordinates['pmid'].isin(pmids)]


def which_pmids(tfidf, pmids_iter, keywords_iter, keyword):
    """Return the publications' id related to the given keyword.

    Parameters
    ----------
    tfidf : matrix like
        Matrix which columns correspond to the keywords and lines
        correspond to the publications' ids.
    pmids_iter : iterable or string
        Iterable over the available publications' id or path to file containing
        them.
    keywords_iter : iterable or string
        Iterable over the available keywords or path to file containing them.
    keyword : string
        Keyword

    Returns
    -------
    list
        list of publications' ids related to the keyword according to the tfidf
        matrix.
    """
    _, decode_pmid = build_index(pmids_iter)
    encode_keyword, _ = build_index(keywords_iter)

    nz_encoded_pmids = tfidf[:, encode_keyword[keyword]].nonzero()[0]
    return [int(decode_pmid[idx]) for idx in nz_encoded_pmids]
