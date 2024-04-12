import sys
import yaml

import re
import langdetect
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# config_file="/Users/zarreen/Documents/HOLT/holt-final-delivery/config.yml"
def load_config(config_file=None):

    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config
    
    
def clean_language(series: pd.Series, langs: list):
    """
    Pipeline step for cleaning unwanted languages by column using langdetect.
    See langdetect documentation for supported languages, generally in 2 letter form
    Ex: ['en', fr']

    :param series: Pandas Series (DataFrame column)
    :param langs: List of accepted languages
    :return: Pandas Series (DataFrame column)
    """

    def __language_detect(x):
        try:
            lang = langdetect.detect(x)
            return lang if lang else ''
        except langdetect.LangDetectException:
            return 'LANGDETECTERROR'

    series = series[series.apply(__language_detect).isin(langs)]
    return series


def lowercase(series: pd.Series):
    """
    Pipeline step for transforming all text in lowercase

    :param series: Pandas Series (DataFrame column)
    :return: Pandas Series (DataFrame column)
    """

    series = series.apply(str.lower)
    return series


def tokenize_emails(series: pd.Series):
    """
    Pipeline step for tokenizing emails. Words matching used regex
    will be replaced with 'EMAILTOKEN'

    :param series: Pandas Series (DataFrame column)
    :return: Pandas Series (DataFrame column)
    """

    series = series.str.replace(r'^\w+[._]?\w+@\w+[.]\w{2,3}$',
                                'EMAILTOKEN', regex=True)
    return series


def tokenize_websites(series: pd.Series):
    """
    Pipeline step for tokenizing websites. Words matching used regex
    will be replaced with 'WEBSITETOKEN'

    :param series: Pandas Series (DataFrame column)
    :return: Pandas Series (DataFrame column)
    """

    series = series.str.replace(r'((http|https)://)?[\w./?:@\-_=#]+\.([a-z]){2,4}([\w.&/?:@\-_=#])*',
                                ' WEBSITETOKEN ', regex=True)
    return series


def tokenize_phone_numbers(series: pd.Series):
    """
    Pipeline step for tokenizing phone numbers. Words matching used regex
    will be replaced with 'PHONENUMBERTOKEN'

    :param series: Pandas Series (DataFrame column)
    :return: Pandas Series (DataFrame column)
    """

    series = series.str.replace(r'(\d{3}[-.\s]??\d{3}[-.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-.\s]??\d{4}|\d{3}[-.\s]??\d{4})',
                                ' PHONENUMBERTOKEN ', regex=True)
    return series


def tokenize_percentages(series: pd.Series):
    """
    Pipeline step for tokenizing phone numbers. Words matching used regex
    will be replaced with 'PERCENTAGETOKEN'

    :param series: Pandas Series (DataFrame column)
    :return: Pandas Series (DataFrame column)
    """

    series = series.str.replace(r'(\d+(\.\d+)?\s?%)|(%\s?\d+(\.\d+)?)',
                                ' PERCENTAGETOKEN ', regex=True)
    return series


def tokenize_money(series: pd.Series):
    """
    Pipeline step for tokenizing phone numbers. Words matching used regex
    will be replaced with 'MONEYTOKEN'

    :param series: Pandas Series (DataFrame column)
    :return: Pandas Series (DataFrame column)
    """

    series = series.str.replace(r'(\d+(\.\d{0,2})?\s?\$)|(\$\s?\d{0,2}(\.\d+)?)',
                                ' MONEYTOKEN ', regex=True)

    return series


def tokenize_number_words(series: pd.Series):
    """
    Pipeline step for tokenizing words containing numbers. Words matching used regex
    will be replaced with 'NUMBERWORDTOKEN'

    :param series: Pandas Series (DataFrame column)
    :return: Pandas Series (DataFrame column)
    """

    series = series.str.replace(r'([A-Za-z]+\d+\w*|\d+[A-Za-z]+\w*)',
                                ' NUMBERWORDTOKEN ', regex=True)
    return series


def tokenize_numbers(series: pd.Series):
    """
    Pipeline step for tokenizing numbers. Words matching used regex
    will be replaced with 'NUMBERTOKEN'

    :param series: Pandas Series (DataFrame column)
    :return: Pandas Series (DataFrame column)
    """

    series = series.str.replace(r'(\b\d+\b(\.\b\d+\b)?)',
                                ' NUMBERTOKEN ', regex=True)
    return series


def rm_non_alphanumerical(series: pd.Series):
    """
    Pipeline step for removing all non-alphanumerical characters.
    Characters are replaced with a blank space

    :param series: Pandas Series (DataFrame column)
    :return: Pandas Series (DataFrame column)
    """

    series = series.str.replace(r'[^ \w+]',
                                ' ', regex=True)
    return series


def rm_char_words(series: pd.Series):
    """
    Pipeline step for removing single character words

    :param series: Pandas Series (DataFrame column)
    :return: Pandas Series (DataFrame column)
    """

    series = series.str.replace(r'^[^ ]*',
                                ' ', regex=True)
    return series


def rm_stopwords(series: pd.Series, langs: list, conversion_dict: dict = None):
    """
    Pipeline step for removing stopwords from provided languages,
    given by nltk.corpus.stopwords. I hardcoded a small conversion dictionary
    between langdetect and nltk formats for English, French, Spanish and Italian,
    but if needed, input something else

    :param conversion_dict: Dict
    :param langs: List
    :param series: Pandas Series (DataFrame column)
    :return: Pandas Series (DataFrame column)
    """

    if not conversion_dict:
        conversion_dict = {'en': 'english',
                           'fr': 'french',
                           'es': 'spanish',
                           'it': 'italian'}
    stop_langs = [conversion_dict[lang] for lang in langs]
    stopwords = {r'\b' + re.escape(stop) + r'\b': '' for lang in stop_langs for stop in nltk_stopwords.words(lang)}
    series = series.replace(stopwords, regex=True)
    return series


def clean_whitespaces(series: pd.Series):
    """
    Pipeline step for cleaning multiple consecutive spaces
    and other types of whitespaces

    :param series: Pandas Series (DataFrame column)
    :return: Pandas Series (DataFrame column)
    """
    series = series.str.replace("\n", " ")
    series = series.str.replace("\t", " ")
    series = series.str.replace(r'\s+', ' ', regex=True)
    series = series.str.strip()
    return series


def lemmatize(series: pd.Series):
    """
    Pipeline step for lemmatizing all words in a column,
    as implemented by nltk.stem.WordNetLemmatizer

    :param series: Pandas Series (DataFrame column)
    :return: Pandas Series (DataFrame column)
    """
    lm = WordNetLemmatizer()

    def __lemmatize_text(text: str):
        return ' '.join([lm.lemmatize(w) for w in text.split()])

    series = series.apply(__lemmatize_text)
    return series


def rm_banlist(series: pd.Series, banlist: list):
    """
    Pipeline step for removing all occurrences of words in
    the provided banlist

    :param banlist: List
    :param series: Pandas Series (DataFrame column)
    :return: Pandas Series (DataFrame column)
    """

    banlist = {r'\b' + re.escape(w) + r'\b': '' for w in banlist}
    series = series.replace(banlist, regex=True)
    series = series.str.strip()
    return series


def count_occurrences(df: pd.DataFrame, columns: list):
    """
    Count number of occurrences of each type of word in provided dataframe
    on provided columns, returning a dataframe containing relevant
    statistical information

    :param columns: List
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """

    # Collect all cells into string
    collection = [df[col][row] for row in df[columns].index for col in columns]
    collection = [str(s) for s in collection if s == s and s is not None and str(s).strip() != '']
    collected_text = re.sub('\n', ' ', ' '.join(collection))

    # Count occurrences
    d = {}
    for word in collected_text.split():
        try:
            d[word] += 1
        except KeyError:
            d[word] = 1

    # Create dataframe
    count_df = pd.DataFrame.from_dict(d, orient='index').sort_values(0, ascending=False)

    # Cumulative sum and percentages
    count_df['cum_sum'] = count_df[0].cumsum()
    count_df['cum_perc'] = 100 * count_df['cum_sum'] / count_df[0].sum()

    return count_df



def remove_tokens(self, string):
    string = string.split()
    cln_series = [s for s in string if not s.endswith("TOKEN")]
    cln_series = " ".join(cln_series)

    return cln_series
