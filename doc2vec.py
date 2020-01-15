import time
import os
import gensim

# Set file names for train and test data
test_data_dir = os.path.join('/home', 'eunwoo', 'Desktop')
lee_train_file = os.path.join(test_data_dir, 'traindata.csv')
lee_test_file = os.path.join(test_data_dir, 'testdata.csv')

from khaiii import KhaiiiApi
def tokenize(sentence):
	token = []
	khaii = KhaiiiApi()
	word_class = ['NNG', 'NNP', 'NNB', 'NR', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'MM', 'MAG', 'MAJ', 'IC', 'XPN', 'XSN', 'XSV', 'XSA', 'XR', 'SN']
	for word in khaii.analyze(sentence):
		for morph in word.morphs:
			if morph.tag in word_class:
				token.append(morph.lex)
	return token

import smart_open
def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            tokens = tokenize(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

start_time = time.time()
train_corpus = list(read_corpus(lee_train_file))
test_corpus = list(read_corpus(lee_test_file, tokens_only=True))
print("khaii time : ", time.time() - start_time)

# Let's take a look at the training corpus
#print(train_corpus[:2])
#print(test_corpus[:2])
start_time = time.time()
model = gensim.models.doc2vec.Doc2Vec(vector_size=52, min_count=2, epochs=40)
model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[1])

import collections

counter = collections.Counter(ranks)

print(counter)

###############################################################################
# Testing the Model
# -----------------
#
# Using the same approach above, we'll infer the vector for a randomly chosen
# test document, and compare the document to our model by eye.
#
import random
# Pick a random document from the test corpus and infer a vector from the model
doc_id = random.randint(0, len(test_corpus) - 1)
inferred_vector = model.infer_vector(test_corpus[doc_id])
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

# Compare and print the most/median/least similar documents from the train corpus
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
print(u'%s %s: «%s»\n' % ('MOST', sims[0], ' '.join(train_corpus[sims[0][0]].words)))

print("model time : ", time.time() - start_time)