import argparse
import nltk
from nltk.collocations import *
import sys
import pandas as pd
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus_path", default=None, type=str, required=True,
                        help="path to the corpus text files .txt")
    parser.add_argument("--corpus_name", default=None, type=str, required=True,
                        help="the name of hte corpus")
    parser.add_argument("--wanted_word1", default=None, type=str, required=True,
                        help="te word to look for in the colloction")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="Output directory to save the results")
    
    args = parser.parse_args()
    print(args)
    corpus_path = args.corpus_path
    corpus_name = args.corpus_name
    wanted_word1 = args.wanted_word1
    output_dir = args.output_dir


    print("read files")
    corpus_file = open(corpus_path, 'r').read()
    corpus_list = corpus_file.split()
    print("load corpus")
    corpus = nltk.Text(corpus_list)
    print(len(corpus))
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    word1 = []
    word2 = []
    pmi = []


    ignored_words = nltk.corpus.stopwords.words('english')
    ## Bigrams
    finder = BigramCollocationFinder.from_words(corpus)
    # only bigrams that appear 3+ times
    finder.apply_freq_filter(0)
    # only bigrams that contain 'creature'
    finder.apply_ngram_filter(lambda *w: wanted_word1 not in w)
    finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
    for i in finder.score_ngrams(bigram_measures.pmi):
        word1.append(i[0][0])
        word2.append(i[0][1])
        pmi.append(i[1])

    pd.DataFrame({"word1":word1, "word2":word2, "PMI":pmi}).to_csv(output_dir+"/"+corpus_name+"_"+wanted_word1+"_collocation.csv")

if __name__ == "__main__":
    main()




