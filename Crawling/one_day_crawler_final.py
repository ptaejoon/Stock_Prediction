#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd 
import numpy as np

import pymysql
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()

import html5lib
from datetime import datetime


# In[8]:


# db에서 종목코드 가져오기
# db 추출 연결
#종목에서 끌어올 코드번호 데이터 프레임에 삽입

db = pymysql.connect(host = 'sp-articledb.clwrfz92pdul.ap-northeast-2.rds.amazonaws.com', port = 3306,user ='admin',passwd = 'sogangsp',db='mydb',charset = 'utf8')
cursor = db.cursor()

sql = """
            select * from CODE_INFORMATION ;
        """
cursor.execute(sql)
db.commit()

code_data = pd.DataFrame(cursor.fetchall(), columns=['code', 'name'])

#str 형태로 저장되어있는 코드명을 int로 치환 후 6자리 int형으로 고정
code_data.code = code_data.code.astype(int)
code_data.code=code_data.code.map('{:06d}'.format)


# In[9]:



def get_url(item_name, code_data): 
    code = code_data.query("name=='{}'".format(item_name))['code'].to_string(index=False)
    code = code.strip(' ')
    url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)

    print("요청 URL = {}".format(url)) 
    return url

def crawler_chart(item_name) :
    url = get_url(item_name, code_data) 

    # 일자 데이터를 담을 df라는 DataFrame 정의 
    df = pd.DataFrame() 

    for page in range(1, 2): 
        pg_url = '{url}&page={page}'.format(url=url, page=page) 
        df = df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True)

    df = df.dropna()
    
    return df 

def oneday_crawl(item_name) :
    update_chart = pd.DataFrame(columns=['CORP_NAME','TRADE_TIME','CLOSE_PRICE','DIFF','OPEN_PRICE','MAX_PRICE','MIN_PRICE','TRADE_AMOUNT'])
    now = datetime.now()
    for i in crawler_chart(item_name).get_values() :
        if i[0] == '{:04d}.{:02d}.{:02d}'.format(now.year,now.month,now.day) :
            data = {
                'CORP_NAME' : item_name
                ,'TRADE_TIME' : i[0]
                ,'CLOSE_PRICE' : int(i[1])
                ,'DIFF' : int(i[2])
                ,'OPEN_PRICE' : int(i[3])
                ,'MAX_PRICE' : int(i[4])
                ,'MIN_PRICE' : int(i[5])
                ,'TRADE_AMOUNT' : int(i[6])
            }
            update_chart=update_chart.append(data,ignore_index=True)
    
    return update_chart


# In[10]:


update_chart = pd.DataFrame(columns=['CORP_NAME','TRADE_TIME','CLOSE_PRICE','DIFF','OPEN_PRICE','MAX_PRICE','MIN_PRICE','TRADE_AMOUNT'])
for i in code_data.name :
    update_chart=update_chart.append(oneday_crawl(i))


# In[330]:


# 오늘 데이터 삽입
engine = create_engine("mysql+pymysql://admin:"+"sogangsp"+"@sp-articledb.clwrfz92pdul.ap-northeast-2.rds.amazonaws.com:3306/mydb?charset=utf8", encoding='utf-8')
conn = engine.connect()
update_chart.to_sql(name = 'CORP_STOCK',con = engine, if_exists = 'append',index = False)


# In[ ]:




