from khaiii import KhaiiiApi

def tokenize(sentence):
	token = []
	khaii = KhaiiiApi()
	for word in khaii.analyze(sentence):
		for morph in word.morphs:
			token.append(morph.lex)
	return token
