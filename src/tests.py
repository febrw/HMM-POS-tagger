from nltk import FreqDist, WittenBellProbDist
from io import open
from conllu import parse_incr
from math import log2, inf
import p1

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
lang = 'en'

train_sents = conllu_corpus(train_corpus(lang))
test_sents = conllu_corpus(test_corpus(lang))

#---------------------------------------------------
# TESTS
#---------------------------------------------------

def info_test():
	print(len(train_sents), len(test_sents)) # 6911, 1096

def generate_tags_test():
	tags = list(set([w['upos'] for x in train_sents for w in x]))
	print(tags)

def generate_transitions():
	all_transitions = []
	for sentence in train_sents:
		tags = ['<s>'] + [token['upos'] for token in sentence] + ['</s>']
		transitions_per_sentence = list(zip(tags[:-1], tags[1:]))
		all_transitions += transitions_per_sentence
	return all_transitions

def generate_transitions_test():
	info = []
	for sentence in train_sents:
		words = [token['form'] for token in sentence]
		raw_tags = [token['upos'] for token in sentence]
		tags = ['<s>'] + [token['upos'] for token in sentence] + ['</s>']
		transitions = list(zip(tags[:-1], tags[1:]))
		sent_info = (words, raw_tags, transitions)
		info.append(sent_info)

	for words, raw_tags, transitions in info[:5]:
		print(words)
		print(raw_tags)
		print(transitions)
		print()

def transition_model():
	transitions = generate_transitions()
	prior_tags = set([t for (t,_) in transitions])
	transition_model = {}
	for prior_tag in prior_tags:
		observed_tags = [observed for (prior, observed) in transitions if prior == prior_tag]
		transition_model[prior_tag] = WittenBellProbDist(FreqDist(observed_tags), bins = 1e6)
	return transition_model


def viterbi_decoding_test():
	hmm = p1.HMM('en')
	observations = "Will can spot Mary".split()
	viterbi_matrix = [{tag : 0 for tag in hmm.tags} for _ in observations]
	back_pointers = ['' for _ in observations] # stores tags, which are used as dict keys for each observation

	#print(len(viterbi_matrix[0]))
	#print(len(hmm.tags))

	print("Row: ", observations[0])
	for tag in hmm.tags:
		viterbi_matrix[0][tag] = hmm.get_transition('<s>', tag) + hmm.get_emission(tag, observations[0])
		print('<s>', tag, viterbi_matrix[0][tag])

	# stores POS tag corresponding to highest probability for this emission
	back_pointers[0] = max(viterbi_matrix[0], key=viterbi_matrix[0].get)
	print()
	print('ML tag: ', back_pointers[0])
	print()

	for index, observation in enumerate(observations[1:], 1):
		print("Row: ", observation)
		for tag in hmm.tags:
			prior_tag = back_pointers[index-1]
			viterbi_matrix[index][tag] = viterbi_matrix[index-1][prior_tag] + hmm.get_transition(prior_tag, tag) + hmm.get_emission(tag, observation)
			print(prior_tag, tag, viterbi_matrix[index][tag])
		back_pointers[index] = max(viterbi_matrix[index], key=viterbi_matrix[index].get)
		print()
		print('ML tag: ', back_pointers[index])
		print()
	terminal_prob = viterbi_matrix[-1][back_pointers[-1]] + hmm.get_transition(back_pointers[-1], '</s>')

	print(back_pointers)

def test_get_results():
	hmm = p1.HMM('en')
	test_tag_sequences = [ [token['upos'] for token in sentence] for sentence in hmm.test_sents]
	for i in range(5):
		print(' '.join([x['form'] for x in hmm.test_sents[i]]))
		print("real: ", test_tag_sequences[i])
		print("mine: ", hmm.viterbi_decoding([x['form'] for x in hmm.test_sents[i]])[0])
		print()

def test_special_tag_emissions():
	hmm = p1.HMM('en')
	for word in set([x['form'] for sentence in train_sents for x in sentence]):
		print(hmm.get_emission('<s>', word))

def start_end_tags_test():
	hmm = p1.HMM()


def test_tag_lengths():
	bad_sentences = [sentence for sentence in train_sents if any([token['upos']=='X' for token in sentence])]
	#print(len(train_sents))
	#print(len(bad_sentences))

	for sentence in bad_sentences[:5]:
		readable = ' '.join([x['form'] for x in sentence])
		tags = [x['upos'] for x in sentence]
		#print(readable)
		#print(tags)
		#print()

	word_tags = [(x['form'], x['upos']) for sentence in train_sents for x in sentence]
	print(len(word_tags))

	x_tags = [p for p in word_tags if p[1] == 'X']
	print(len(x_tags))
	for w,t in x_tags:
		print("unk word: ", w)
		print(t)
		print()


def main():
	freqs = dict(FreqDist([token['form'] for sentence in train_sents for token in sentence]))
	hapax = set([word for word in freqs if freqs[word] < 2])

	for i in hapax:
		print(i)

	print(len(hapax))



if __name__ == '__main__':
	main()
