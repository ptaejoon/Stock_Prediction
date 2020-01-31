import time
import os
import gensim
import pymysql
from khaiii import KhaiiiApi
import sys
import gc

db_total = 5000000 # per a DB
db_step = 10000
db_iter = db_total // db_step
db_start = 2980000 # point where to start
print("model load start!")
model = gensim.models.doc2vec.Doc2Vec.load('vocab.model') # make a doc2vec model
print("model load finish!")
# khaiii API function
khaii = KhaiiiApi()
def tokenize(sentence):
	token = []
	word_class = ['NNG', 'NNP', 'NNB', 'NR', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'MM', 'MAG', 'MAJ', 'IC', 'XPN', 'XSN', 'XSV', 'XSA', 'XR', 'SN']
	for word in khaii.analyze(sentence):
		for morph in word.morphs:
			if morph.tag in word_class:
				token.append(morph.lex)
	return token

# Eunwoo DB connection
print("Eunwoo DB Start!")
conn = pymysql.connect(host='sp-articledb.clwrfz92pdul.ap-northeast-2.rds.amazonaws.com', user = 'admin', password='sogangsp', db='mydb', charset='utf8', port=3306)
curs = conn.cursor()
for i in range(db_start // db_step, db_iter):
    sql = "select news from article where newsid between " + str(i * db_step + 1) + " and " + str((i+1) * db_step)
    curs.execute(sql)
    rows = curs.fetchall()
    arr = dict()
    print(str(i)+'th')
    for index, row in enumerate(rows, 1) :
        try:
            tokens = tokenize(row[0])
            for token in tokens:
                if token in arr:
                    arr[token] += 1
                else:
                    arr[token] = 1
            if index % (db_step//10) == 0:
                print("db(", str(i * db_step + index), ") is done")
                print("tokens", sys.getsizeof(tokens), " / ", id(tokens))
                print("arr", sys.getsizeof(arr))
                print("khaiii", sys.getsizeof(khaii))
                gc.collect()
        except KeyboardInterrupt:
            exit()
        except:
            print("Khaiii has problem")
            continue
    if i != 0:
        model.build_vocab_from_freq(arr, update = True)
    else:
        model.build_vocab_from_freq(arr)
    model.save('vocab.model')
    print("save is done!(%d)"%((i+1) * db_step))
conn.close()
print(len(model.wv.vocab))
exit()
# Soohwan DB connection
print("Soohwan DB Start")
conn = pymysql.connect(host='article-raw-data.cnseysfqrlcj.ap-northeast-2.rds.amazonaws.com', user = 'admin', password='sogangsp', db='mydb', charset='utf8', port=3306)
curs = conn.cursor()
for i in range(db_iter):
    sql = "select news from article where newsid between " + str(i * db_step) + " and " + str((i+1) * db_step)
    curs.execute(sql)
    rows = curs.fetchall()
    arr = dict()

    print("db(", db_step * i, ") start")
    for row in rows :
        tokens = tokenize(row[0])
        for token in tokens:
            if token in arr:
                arr[token] += 1
            else:
                arr[token] = 1
    model.build_vocab_from_freq(arr)
    model.save('vocab.model')
conn.close()
