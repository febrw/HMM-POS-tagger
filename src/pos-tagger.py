from nltk import FreqDist, WittenBellProbDist
from io import open
from conllu import parse_incr
from math import log2, inf
import preprocessor

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

# Choose language.

#lang = 'en'

#train_set = conllu_corpus(train_corpus(lang))
#test_set = conllu_corpus(test_corpus(lang))



#----------------------------------
# HMM class for viterbi algorithm
#----------------------------------

class HMM:
	
	def __init__(self, lang, preprocess=False):
		
		if preprocess:
			pp = preprocessor.Preprocessor(conllu_corpus(train_corpus(lang)))
			if lang == 'en':
				self.train_sents = pp.augment_eng_train_sents(conllu_corpus(train_corpus(lang)))
				self.test_sents = pp.augment_eng_test_sents(conllu_corpus(test_corpus(lang)))
			elif lang == 'fr':
				self.train_sents = pp.augment_french_train_sents(conllu_corpus(train_corpus(lang)))
				self.test_sents = pp.augment_french_test_sents(conllu_corpus(test_corpus(lang)))
			elif lang == 'uk':
				self.train_sents = pp.augment_ukr_train_sents(conllu_corpus(train_corpus(lang)))
				self.test_sents = pp.augment_ukr_test_sents(conllu_corpus(test_corpus(lang)))
		else:
			self.train_sents = conllu_corpus(train_corpus(lang))
			self.test_sents = conllu_corpus(test_corpus(lang))
		self.tags = list(set([w['upos'] for x in self.train_sents for w in x]))
		self.transition_model = {}
		self.emission_model = {}

		self.train_transition_model()
		self.train_emission_model()
		

	# returns a list of pairs: (prior tag, observed tag)
	def generate_transitions(self):
		all_transitions = []
		for sentence in self.train_sents:
			tags = ['<s>'] + [token['upos'] for token in sentence] + ['</s>']
			transitions_per_sentence = list(zip(tags[:-1], tags[1:]))
			all_transitions += transitions_per_sentence
		return all_transitions

	# returns a list of pairs: (tag, word)
	def generate_emissions(self):
		return [(token['upos'], token['form']) for sentence in self.train_sents for token in sentence]
	

	def train_transition_model(self):
		transitions = self.generate_transitions()
		prior_tags = set([t for (t,_) in transitions])
		for prior_tag in prior_tags:
			observed_tags = [observed for (prior, observed) in transitions if prior == prior_tag]
			self.transition_model[prior_tag] = WittenBellProbDist(FreqDist(observed_tags), bins = 1e5)
		return
	
	def train_emission_model(self):
		emissions = self.generate_emissions()
		tags = set([t for (t,_) in emissions])
		for tag in tags:
			words = [w for (t,w) in emissions if t == tag]
			self.emission_model[tag] = WittenBellProbDist(FreqDist(words), bins = 1e5)
		return
	
	def get_emission(self, tag, word):
		return log2(self.emission_model[tag].prob(word))
	
	def get_transition(self, prior_tag, observed_tag):
		return log2(self.transition_model[prior_tag].prob(observed_tag))
	

	def viterbi_decode(self, observations):
		viterbi_matrix = [{tag : 0 for tag in self.tags} for _ in observations]
		back_pointers = [{tag : '' for tag in self.tags} for _ in observations] # stores tags, which are used as dict keys for each observation

		# initialisation
		for tag in self.tags:
			viterbi_matrix[0][tag] = self.get_transition('<s>', tag) + self.get_emission(tag, observations[0])

		# recursive step
		for index, observation in enumerate(observations[1:], 1):
			for tag in self.tags:
				potential_probs = [(viterbi_matrix[index-1][prior_tag] + self.get_transition(prior_tag, tag) + self.get_emission(tag, observation), prior_tag) for prior_tag in self.tags]
				max_prob, arg_max = max(potential_probs, key=lambda i:i[0])
				viterbi_matrix[index][tag] = max_prob
				back_pointers[index][tag] = arg_max

		# termination step
		potential_probs = [(viterbi_matrix[-1][prior_tag] + self.get_transition(prior_tag, '</s>'), prior_tag) for prior_tag in self.tags]
		terminal_prob, arg_max = max(potential_probs, key=lambda i:i[0])
		viterbi_tag_sequence = []
		for i in range(len(observations) - 1, -1, -1):
			viterbi_tag_sequence.insert(0, arg_max)
			arg_max = back_pointers[i][arg_max]

		return viterbi_tag_sequence, terminal_prob

	def get_viterbi_accuracy(self):
		test_tag_sequences = [ token['upos'] for sentence in self.test_sents for token in sentence]
		output_tag_sequences = [tag for sentence in self.test_sents for tag in self.viterbi_decode([token['form'] for token in sentence])[0]]
		accuracy = sum(ground_truth == prediction for ground_truth, prediction in zip(test_tag_sequences, output_tag_sequences)) / len(test_tag_sequences)
		return accuracy

def print_all_viterbi_results():
	languages = {'en' : 'English: ', 'fr' : 'French: ', 'uk' : 'Ukrainian: '}
	for lang in languages:
		hmm = HMM(lang, preprocess=False)
		print('Accuracy for ' + languages[lang] + str(round(hmm.get_viterbi_accuracy()*100, 2)) + '%')		
	return

def print_all_preprocessed_viterbi_results():
	languages = {'en' : 'English: ', 'fr' : 'French: ', 'uk' : 'Ukrainian: '}
	for lang in languages:
		hmm = HMM(lang, preprocess=True)
		print('Accuracy for ' + languages[lang] + str(round(hmm.get_viterbi_accuracy()*100, 2)) + '%')		
	return

def test_model():
	hmm = HMM('uk', preprocess=True)
	print('Accuracy for Ukrainian: ' + str(round(hmm.get_viterbi_accuracy()*100, 2)) + '%')

def main():
	print_all_viterbi_results()
	#print_all_preprocessed_viterbi_results()

if __name__ == '__main__':
	main()