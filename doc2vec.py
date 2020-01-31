import time
import os
import gensim
from khaiii import KhaiiiApi
import pymysql
import time

# Set file names for train and test data


khaii = KhaiiiApi()
word_class = ['NNG', 'NNP', 'NNB', 'NR', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'MM', 'MAG', 'MAJ', 'IC', 'XPN', 'XSN', 'XSV', 'XSA', 'XR', 'SN']
def tokenize(sentence):
    token = []
    for word in khaii.analyze(sentence):
        for morph in word.morphs:
            if morph.tag in word_class:
                token.append(morph.lex)
    return token

conn = pymysql.connect(host='localhost', user = 'sinunu', password='1q2w3e4r', db='mydb', charset='utf8')
curs = conn.cursor()

model = gensim.models.doc2vec.Doc2Vec.load('vocab_20.model')

#even_article table start
print("even_article start")
for i in range(1, 680):
    sql = "select news from even_article where newsid between " + str(i*10000) + " and " + str( (i+1)*10000-1)
    curs.execute(sql)
    rows = curs.fetchall()
    arr = []
    for index, row in enumerate(rows):
        try:
            tokens = tokenize(row[0])
            #tokens = gensim.utils.simple_preprocess(row[0])
            arr.append(gensim.models.doc2vec.TaggedDocument(tokens, [i*10000+index]))
        except KeyboardInterrupt:
            exit()
        except:
            print("Khaiii has problem")
            continue
    model.build_vocab(arr, update = True)
    model.save("vocab_20.model")
    print("even article", i*10000, "~", (i+1)*10000-1, "finish")

#odd_article table start
print("odd_article start")
for i in range(700):
    sql = "select news from odd_article where newsid between " + str(i*10000) + " and " + str( (i+1)*10000-1)
    curs.execute(sql)
    rows = curs.fetchall()
    arr = []
    for index, row in enumerate(rows) :
        try:
            tokens = tokenize(row[0])
            arr.append(gensim.models.doc2vec.Doc2Vec.TaggedDocument(tokens,[i*10000+index]))
        except KeyboardInterrupt:
            exit()
        except:
            print("Khaiii has problem")
            continue
    model.build_vocab(arr, update = True)
    model.save("vocab_20.model")
    print("odd article", i*10000, "~", (i+1)*10000-1, "finish")

print("job's done")
conn.close()

#model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)


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
