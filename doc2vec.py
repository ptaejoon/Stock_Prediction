import time
import os
import gensim
from khaiii import KhaiiiApi
import pymysql

# Set file names for train and test data
test_data_dir = os.path.join('/home', 'eunwoo', 'Desktop')
lee_train_file = os.path.join(test_data_dir, 'traindata.csv')
lee_test_file = os.path.join(test_data_dir, 'testdata.csv')

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

conn = pymysql.connect(host='sp-articledb.clwrfz92pdul.ap-northeast-2.rds.amazonaws.com', user = 'admin', password='sogangsp', db='mydb', charset='utf8', port=3306)
curs = conn.cursor()
sql = "select news from article where newsid between 0 and 10000"
curs.execute(sql)
rows = curs.fetchall()
arr = dict()
train_corpus = []
test_corpus = []

for index, row in enumerate(rows) :
    try:
        tokens = tokenize(row[0])
        for token in tokens:
            if token in arr:
                arr[token] += 1
            else:
                arr[token] = 1
    except KeyboardInterrupt:
        exit()
    except:
        print("Khaiii has problem")
        continue

conn.close()

model = gensim.models.doc2vec.Doc2Vec(vector_size=792, min_count=2, epochs=40)
model.build_vocab_from_freq(arr, update = True)

#model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
model.save("vovab_10")
'''
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

start_time = time.time()
print("model time : ", time.time() - start_time)

#model.save("doc2vec.model")
'''