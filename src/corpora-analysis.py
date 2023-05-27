from nltk import FreqDist, WittenBellProbDist
from io import open
from conllu import parse_incr
from math import log2, inf
import matplotlib.pyplot as plt
import numpy as np

treebank = {}
treebank['en'] = '../resources/UD_English-GUM/en_gum'
treebank['fr'] = '../resources/UD_French-Rhapsodie/fr_rhapsodie'
treebank['uk'] = '../resources/UD_Ukrainian-IU/uk_iu'

def train_corpus(lang):
	return treebank[lang] + '-ud-train.conllu'

def test_corpus(lang):
	return treebank[lang] + '-ud-test.conllu'

# Remove contractions such as "isn't".
def prune_sentence(sent):
	return [token for token in sent if type(token['id']) is int]

def conllu_corpus(path):
	data_file = open(path, 'r', encoding='utf-8')
	sents = list(parse_incr(data_file))
	return [prune_sentence(sent) for sent in sents]


#----------------------------------
# Analysis of Corpora
#----------------------------------

langs = {'en' : 'English', 'fr' : 'French', 'uk' : 'Ukrainian'}

def count_hapax_legomena(language):
    words_fd = dict(FreqDist([tok['form'] for sentence in conllu_corpus(train_corpus(language)) for tok in sentence]))
    hapax_legomena = [word for word in words_fd if words_fd[word] < 2]
    return len(hapax_legomena)

def plot_tag_distribution(language):
    tag_freqs = {}
    for sent in conllu_corpus(train_corpus(language)):
        for tok in sent:
            if tok['upos'] in tag_freqs:
                tag_freqs[tok['upos']] += 1
            else:
                tag_freqs[tok['upos']] = 0

    tag_freqs = {k:v for k,v in sorted(tag_freqs.items(), key=lambda i:i[1], reverse=True)}

    plt.bar(range(len(tag_freqs)), list(tag_freqs.values()), align='center')
    plt.xticks(range(len(tag_freqs.keys())),list(tag_freqs.keys()), rotation='vertical')
    plt.yticks(np.arange(0, max(tag_freqs.values()) + 1000, 1000))

    plt.xlabel('Universal POS Tags')
    plt.ylabel('Occurence Frequencies in the ' + langs[language] + ' training corpus')
    plt.show()
    return


def print_hapax_stats(lang):

    fd = dict(FreqDist([tok['form'] for sentence in conllu_corpus(train_corpus(lang)) for tok in sentence]))
    hapax_count = sum([1 for word in fd if fd[word] < 2])
    print("info for lang: ", langs[lang])
    #hapax_count = count_hapax_legomena(lang)
    total = sum([len(x) for x in conllu_corpus(train_corpus('en'))])
    distinct = len(dict(FreqDist([tok['form'] for sentence in conllu_corpus(train_corpus(lang)) for tok in sentence])).keys())

    print(hapax_count, ' hapax count')
    print(total, ' all words count')
    print(distinct, ' num of unique tokens')

    
    print('hapax proportion over word set: ', hapax_count / distinct)
    print('hapax proportion over corpus: ', hapax_count / total)



if __name__ == '__main__':
	main()
