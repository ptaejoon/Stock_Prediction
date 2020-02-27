from tokenizer import tokenize
import pymysql
import sys
from datetime import datetime, timedelta

article = {"host": '127.0.0.1', "port": 3306,
                "user": 'sinunu', "password": '1q2w3e4r', "db": 'mydb', 'charset': 'utf8'}

DBconnect = pymysql.connect(
    host=article["host"],
    port=article['port'],
    user=article['user'],
    password=article['password'],
    db=article['db'],
    charset=article['charset']
)
Cursor = DBconnect.cursor()

InsertDBconnect = pymysql.connect(
    host=article["host"],
    port=article['port'],
    user=article['user'],
    password=article['password'],
    db=article['db'],
    charset=article['charset']
)
InsertCursor = InsertDBconnect.cursor()

odd_load_sql = "select * from odd_article"
even_load_sql = "select * from even_article"
odd_insert_sql = "insert into odd_pv(news,newsid,writetime,section) values (%s,%s,%s,%s)"
even_insert_sql = "insert into even_pv(news,newsid,writetime,section) values (%s,%s,%s,%s)"

Cursor.execute(odd_load_sql)
rows = Cursor.fetchmany(size=100)
while rows:
    for row in rows:
        vec = tokenize(row[0])
        InsertCursor.execute(odd_load_sql%(vec,row[1],row[2],row[3]))
    InsertDBconnect.commit()
    rows = Cursor.fetchmany(size=100)

Cursor.execute(even_load_sql)
rows = Cursor.fetchmany(size=100)
while rows:
    for row in rows:
        vec = tokenize(row[0])
        InsertCursor.execute(even_load_sql%(vec,row[1],row[2],row[3]))
    InsertDBconnect.commit()
    rows = Cursor.fetchmany(size=100)


