import json
import logging
import re
from pathlib import Path

import nltk.test
import pandas as pd
from nltk.corpus.reader import WordListCorpusReader

import portuguese_tagger_processor
from settings import abrevilex

logger = logging.getLogger(__name__)

__output_path = "data/result.json"
__sentilex_path = "sentilex.csv"
__sentilex_index_path = Path("sentilex_index.csv")
__static_files_path = "./static_files"

tweets_path = Path("data/debatenaband/DebateNaBand_1610_1610_tweets_only.csv")


def get_sentiment(x):
    sentiment = x[1].split(";")[3].split("=")[1]
    return int(sentiment)


if __sentilex_index_path.is_file():
    sentilex_index = pd.read_csv(__sentilex_index_path)
else:
    sentilex_index = pd.read_csv(__sentilex_path)
    sentilex_index["sentiment"] = sentilex_index.apply(lambda row: get_sentiment(row), axis=1)
    sentilex_index = sentilex_index.drop(columns=["pos"])
    sentilex_index.to_csv(__sentilex_index_path)

if tweets_path.is_file():
    tweets = pd.read_csv(tweets_path)
else:
    raw_data = pd.read_csv("data/debatenaband/DebateNaBand_1610_1610_original.csv")
    tweets = raw_data[["created_at", "conversation_id", "text", "context_annotations"]]
    tweets.to_csv(tweets_path)

stopwords = nltk.corpus.stopwords.words('portuguese')
reader = WordListCorpusReader(__static_files_path, ['symbols.txt'])
symbols = reader.words()
reader = WordListCorpusReader(__static_files_path, ['positive_emoticons.txt'])
positive_emoticons = reader.words()
reader = WordListCorpusReader(__static_files_path, ['negative_emoticons.txt'])
negative_emoticons = reader.words()

tweet_tokenizer = portuguese_tagger_processor.get_tweet_tokenizer()
tagger = portuguese_tagger_processor.get_tagger()
json_result = []


def count_positive_emoticons(tokens):
    counter = 0
    for emoticon in positive_emoticons:
        if emoticon in tokens:
            counter += 1
    return counter


def count_negative_emoticons(tokens):
    counter = 0
    for emoticon in negative_emoticons:
        if emoticon in tokens:
            counter += 1
    return counter


def replace_symbols(text):
    for symbol in symbols:
        text = text.replace(symbol, "")
    return text


def replace_urls(text):
    return re.sub(r"http\S+", "", text)


def replace_abbreviations(tokens):
    for abbreviation in abrevilex.keys():
        if abbreviation in tokens:
            i = tokens.index(abbreviation)
            tokens[i] = abrevilex[abbreviation]
    return tokens


def remove_stopwords(tokens):
    for stopword in stopwords:
        if stopword in tokens:
            tokens.remove(stopword)
    return tokens


def remove_symbols(tokens):
    try:
        for symbol in symbols:
            if symbol in tokens:
                if symbol == "...":
                    tokens[symbol] = " "
                else:
                    tokens.remove(symbol)
    except Exception:
        pass
    return tokens


def text_processor(tweet):
    text = tweet.lower()
    text = replace_urls(text)

    if "rt" in text:
        try:
            text = replace_symbols(text.split(":")[1])
        except IndexError:
            pass
    else:
        text = replace_symbols(text)

    return text.strip()


def tokens_processor(tokens):
    tokens = remove_stopwords(tokens)
    tokens = remove_symbols(tokens)
    tokens = replace_abbreviations(tokens)
    return tokens


def sentiments_processor(row, text, tokens, tags):
    adjectives = []
    adverbs = []
    for tagged_word in tags:
        word = tagged_word[0]
        tag = tagged_word[1]
        if tag == "ADJ":
            adjectives.append(word)
        if tag == "ADV" or tag == "ADVL+adv":
            adverbs.append(word)

    if len(adjectives) > 0:
        positive = 0
        negative = 0
        for adjective in adjectives:
            sentiment = None
            try:
                sentiment = sentilex_index.loc[sentilex_index['adjetivo'] == adjective, 'sentiment'].iloc[0]
            except IndexError:
                pass
            has_negative_adverbs = "não" in adverbs
            if sentiment:
                if has_negative_adverbs:
                    if sentiment == 1:
                        negative += 1
                    elif sentiment == -1:
                        positive += negative
                else:
                    if sentiment == 1:
                        positive += 1
                    elif sentiment == -1:
                        negative += negative

        positive = positive + count_positive_emoticons(tokens)
        negative = negative + count_negative_emoticons(tokens)

        score = 0
        sum = positive + negative
        dif = positive - negative

        has_jair = "jair" in row["context_annotations"].lower().strip()
        has_lula = "lula" in row["context_annotations"].lower().strip()

        if sum > 0:
            score = dif / sum

        if score > 0.5:
            json_result.append(
                    {
                        "label": 'positive',
                        'processed_text': text,
                        'text': row["text"],
                        'created_at': row['created_at'],
                        'conversation_id': row['conversation_id'],
                        'has_lula': has_lula,
                        'has_jair': has_jair
                    }
            )
        elif score < 0.5:
            json_result.append(
                    {
                        "label": 'negative',
                        'processed_text': text,
                        'text': row["text"],
                        'created_at': row['created_at'],
                        'conversation_id': row['conversation_id'],
                        'has_lula': has_lula,
                        'has_jair': has_jair
                    }
            )
        else:
            json_result.append(
                    {
                        "label": 'neutral',
                        'processed_text': text,
                        'text': row["text"],
                        'created_at': row['created_at'],
                        'conversation_id': row['conversation_id'],
                        'has_lula': has_lula,
                        'has_jair': has_jair
                    }
            )


def tweet_processor(row):
    text = text_processor(row["text"])
    tokens = tweet_tokenizer.tokenize(text)
    tokens = tokens_processor(tokens)
    tags = tagger.tag(tokens)
    sentiments_processor(row, text, tokens, tags)


def save():
    json_file = open(__output_path, 'w')
    json.dump(json_result, json_file, indent=4)
    json_file.close()


if __name__ == '__main__':
    nltk.download('stopwords')
    tweets['sentiment'] = tweets.apply(lambda row: tweet_processor(row), axis=1)
    save()
