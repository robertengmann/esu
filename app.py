import sys
from keyword_extraction import KeywordExtractor


def main(argv):
    corpus = open(argv[0], 'r').read()
    keyword_extractor = KeywordExtractor(corpus)
    keywords = keyword_extractor.extract_keywords()

    print("\n".join(keywords))


if __name__ == "__main__":
    main(sys.argv[1:])
