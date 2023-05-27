#--------------------------------------
# Tag Augmenter
# provides functions which augment tags
#--------------------------------------
from nltk import FreqDist


class Preprocessor:

	def __init__(self, train_sentences):
		word_freqs = dict(FreqDist([token['form'] for sentence in train_sentences for token in sentence]))
		self.infrequent_words = set([word for word in word_freqs if word_freqs[word] < 2])
		self.frequent_words = [word for word in word_freqs if word not in self.infrequent_words]

	def augment_eng_train_sents(self, sentences):
		for sentence in sentences:
			for token in sentence:
				if token['form'] in self.infrequent_words:
					if token['form'][0].isupper():
						token['form'] = 'UNK-cap'
					elif token['form'].endswith('ing'):
						token['form'] = 'UNK-ing'
					elif token['form'].endswith('ly'):
						token['form'] = 'UNK-ly'
					elif token['form'].endswith('al'):
						token['form'] = 'UNK-al'
					elif token['form'].endswith('ble'):
						token['form'] = 'UNK-ble'
					elif token['form'].endswith('ion'):
						token['form'] = 'UNK-ion'
					elif '@' in token['form']:
						token['form'] = 'UNK-email'
					elif token['form'].endswith('ed'):
						token['form'] = 'UNK-ed'
					elif token['form'].endswith('tive'):
						token['form'] = 'UNK-tive'
					elif token['form'].endswith('ate'):
						token['form'] = 'UNK-ate'
					elif not token['form'].isascii():
						token['form'] = 'UNK-x'
					#elif token['form'].endswith('ions'):
						#token['form'] = 'UNK-ions'
					#elif token['form'][0] == '+' and sum([1 for x in token['form'][1:] if x.isnumeric()]) == 10:
						#token['form'] = 'UNK-phone'
					#elif token['form'].startswith('de'):
					#	token['form'] = 'de-UNK'
					#elif token['form'].endswith('\'s'):
					#	token['form'] = 'UNK-\'s'
					#elif token['form'].endswith('lity'):
					#	token['form'] = 'UNK-lity'
					#elif token['form'].startswith('un'):
					#	token['form'] = 'un-UNK'
					#elif token['form'].startswith('il'):
						#token['form'] = 'il-UNK'
					#elif token['form'].startswith('im'):
						#token['form'] = 'im-UNK'
					#elif token['form'].startswith('in'):
						#token['form'] = 'in-UNK'
					#else:
						#token['form'] = 'UNK'
		return sentences

	def augment_eng_test_sents(self, sentences):
		for sentence in sentences:
			for token in sentence:
				if token['form'] not in self.frequent_words:
					if token['form'][0].isupper():
						token['form'] = 'UNK-cap'
					elif token['form'].endswith('ing'):
						token['form'] = 'UNK-ing'
					elif token['form'].endswith('ly'):
						token['form'] = 'UNK-ly'
					elif token['form'].endswith('al'):
						token['form'] = 'UNK-al'
					elif token['form'].endswith('ble'):
						token['form'] = 'UNK-ble'
					elif token['form'].endswith('ion'):
						token['form'] = 'UNK-ion'
					elif '@' in token['form']:
						token['form'] = 'UNK-email'
					elif token['form'].endswith('ed'):
						token['form'] = 'UNK-ed'
					elif token['form'].endswith('tive'):
						token['form'] = 'UNK-tive'
					elif not token['form'].isascii():
						token['form'] = 'UNK-x'
					#elif token['form'].endswith('ions'):
						#token['form'] = 'UNK-ions'
					#elif token['form'][0] == '+' and sum([1 for x in token['form'][1:] if x.isnumeric()]) == 10:
						#token['form'] = 'UNK-phone'
					#elif 'rz' in token['form']
					#elif token['form'].startswith('de'):
						#token['form'] = 'de-UNK'
					#elif token['form'].endswith('\'s'):
						#token['form'] = 'UNK-\'s'
					#elif token['form'].endswith('lity'):
						#token['form'] = 'UNK-lity'
					#elif token['form'].startswith('un'):
						#token['form'] = 'un-UNK'
					#elif token['form'].startswith('il'):
						#token['form'] = 'il-UNK'
					#elif token['form'].startswith('im'):
						#token['form'] = 'im-UNK'
					#elif token['form'].startswith('in'):
						#token['form'] = 'in-UNK'
					#else:
						#token['form'] = 'UNK'
		return sentences

	def augment_french_train_sents(self, sentences):
		for sentence in sentences:
			for token in sentence:
				if token['form'][0].isupper():
					token['form'] = 'UNK-cap'
				elif token['form'].endswith('ment'):
					token['form'] = 'UNK-ment'
				elif token['form'].endswith('tion'):
					token['form'] = 'UNK-tion'
				elif token['form'].endswith('eur'):
					token['form'] = 'UNK-eur'
				elif token['form'].endswith('euse'):
					token['form'] = 'UNK-euse'
				#elif token['form'].endswith('if'):
					#token['form'] = 'UNK-if'
				#elif token['form'].endswith('ance'):
					#token['form'] = 'UNK-ance'
				#elif token['form'].endswith('er'):
					#token['form'] = 'UNK-er'
				
				#elif token['form'].endswith('âtre'):
				#	token['form'] = 'UNK-âtre'
				#if token['form'].endswith('ent'):
				#	token['form'] = 'UNK-ent'
				#if token['form'] in self.infrequent_words:
					#token['form'] = 'UNK'
		return sentences

	def augment_french_test_sents(self, sentences):
		for sentence in sentences:
			for token in sentence:
				if token['form'][0].isupper():
					token['form'] = 'UNK-cap'
				elif token['form'].endswith('ment'):
					token['form'] = 'UNK-ment'
				elif token['form'].endswith('tion'):
					token['form'] = 'UNK-tion'
				elif token['form'].endswith('eur'):
					token['form'] = 'UNK-eur'
				elif token['form'].endswith('euse'):
					token['form'] = 'UNK-euse'
				
				#elif token['form'].endswith('if'):
					#token['form'] = 'UNK-if'
				#elif token['form'].endswith('ance'):
					#token['form'] = 'UNK-ance'
				#elif token['form'].endswith('er'):
					#token['form'] = 'UNK-er'
				#elif token['form'].endswith('tion'):
				#	token['form'] = 'UNK-tion'
				#elif token['form'].endswith('âtre'):
				#	token['form'] = 'UNK-âtre'
				
				#if token['form'].endswith('ent'):
					#token['form'] = 'UNK-ent'
				#if token['form'] not in self.frequent_words:
					#token['form'] = 'UNK'
		return sentences



	def augment_ukr_train_sents(self, sentences):
		for sentence in sentences:
			for token in sentence:
				
				if token['form'].endswith('евий'):
					token['form'] = 'UNK-евий'
				elif token['form'].endswith('овий'):
					token['form'] = 'UNK-овий'
				elif token['form'].endswith('ання'):
					token['form'] = 'UNK-ання'
				elif token['form'].endswith('яння'):
					token['form'] = 'UNK-яння'
				elif token['form'].endswith('ення'):
					token['form'] = 'UNK-ення'
				elif token['form'].endswith('єння'):
					token['form'] = 'UNK-єння'
				elif token['form'].endswith('ство'):
					token['form'] = 'UNK-ство'
				elif token['form'].endswith('цтво'):
					token['form'] = 'UNK-цтво'
				elif token['form'].endswith('ість'):
					token['form'] = 'UNK-ість'
				elif token['form'].endswith('арка'):
					token['form'] = 'UNK-арка'
				

		return sentences

	def augment_ukr_test_sents(self, sentences):
		for sentence in sentences:
			for token in sentence:
				

				if token['form'].endswith('евий'):
					token['form'] = 'UNK-евий'
				elif token['form'].endswith('овий'):
					token['form'] = 'UNK-овий'
				elif token['form'].endswith('ання'):
					token['form'] = 'UNK-ання'
				elif token['form'].endswith('яння'):
					token['form'] = 'UNK-яння'
				elif token['form'].endswith('ення'):
					token['form'] = 'UNK-ення'
				elif token['form'].endswith('єння'):
					token['form'] = 'UNK-єння'
				elif token['form'].endswith('ство'):
					token['form'] = 'UNK-ство'
				elif token['form'].endswith('цтво'):
					token['form'] = 'UNK-цтво'
				elif token['form'].endswith('ість'):
					token['form'] = 'UNK-ість'
				elif token['form'].endswith('арка'):
					token['form'] = 'UNK-арка'
				
		return sentences