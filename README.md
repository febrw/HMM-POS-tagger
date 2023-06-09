# Part Of Speech tagging using Hidden Markov Models

Hidden Markov Model based Part of Speech tagger using the Viterbi Algorithm, implemented in Python. We have experimented with affix substitutions during preprocessing, retaining any positive rules. Rules for English, French, and Ukrainian can be found in preprocessor.py. Core implementation can be found in pos-tagger.py.

HMM tagger accuracies before and after affix substituions:

| Treebank | Before preprocessing acc. (%) | Substitutive preprocessing acc. (%) |
| :---:     |    :----:   |      :---: |
| English | 90.2 | 92.2 |
| French | 91.8 | 93.4 |
| Ukrainian | 85.6 | 86.0 |


Python Libraries:
- CoNLL-U Parser https://pypi.org/project/conllu/
- NLTK https://pypi.org/project/nltk/
