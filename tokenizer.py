from khaiii import KhaiiiApi

khaii = KhaiiiApi()
def tokenize(sentence):
	token = []
	for word in khaii.analyze(sentence):
		for morph in word.morphs:
			token.append(morph.lex)
	return token
