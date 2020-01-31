#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pymysql 
import pandas as pd
import numpy as np
import datetime  


# In[24]:


connection = pymysql.connect(host = 'sp-articledb.clwrfz92pdul.ap-northeast-2.rds.amazonaws.com'
                             ,user ='admin'
                             ,password = 'sogangsp'
                             ,db = 'mydb'
                             ,charset = 'utf8'
                             ,autocommit = True)

cursor = connection.cursor()
sql = 'select * from CODE_INFORMATION  '
cursor.execute(sql)

data = cursor.fetchall()
connection.close()
#data_frame = pd.DataFrame(data,columns=["CORP_NAME","TRADE_TIME","CLOSE_PRICE","DIFF","OPEN_PRICE","MAX_PRICE","MIN_PRICE","TRADE_AMOUNT"])


# In[10]:


start = datetime.datetime(2010,1,4,0,0,0)


# In[11]:


delta = datetime.timedelta(days=i)


# In[12]:


(start + delta).strftime("%Y-%m-%d %H:%M:%S")


# In[65]:


connection = pymysql.connect(host = 'sp-articledb.clwrfz92pdul.ap-northeast-2.rds.amazonaws.com'
                                ,user ='admin'
                                ,password = 'sogangsp'
                                ,db = 'mydb'
                                ,charset = 'utf8'
                                ,autocommit = True)

cursor = connection.cursor()
sql ="select distinct corp_name from CORP_STOCK where TRADE_TIME < '2010-01-05 00:00:00.00' order by CORP_NAME"
cursor.execute(sql)
data = cursor.fetchall()
corp_namelist = pd.DataFrame(data,columns = ["name"])


# In[62]:


df = pd.DataFrame()


# In[63]:


i = 0

while True : 
    delta = datetime.timedelta(days=i)
    
    sql = """select CLOSE_PRICE, OPEN_PRICE, MAX_PRICE, MIN_PRICE, TRADE_AMOUNT
            from CORP_STOCK 
            where TRADE_TIME = "{}" and 
            corp_name in (
            select distinct corp_name from CORP_STOCK 
            where TRADE_TIME < '2010-01-05 00:00:00')
            order by TRADE_TIME, CORP_NAME 
            """.format((start + delta).strftime("%Y-%m-%d %H:%M:%S"))
    
    cursor.execute(sql)
    
    data = np.array(cursor.fetchall())
    if data.size != 0 :
        data_temp = data.flatten(order = 'C')
        df = df.append(pd.DataFrame(data_temp).T)
        print(df)
    i +=1 


# In[ ]:


print(df)


# In[ ]:


connection.close()


# In[ ]:





# In[ ]:


connection = pymysql.connect(host = 'article-raw-data.cnseysfqrlcj.ap-northeast-2.rds.amazonaws.com'
                                ,user ='admin'
                                ,password = 'sogangsp'
                                ,db = 'mydb'
                                ,charset = 'utf8'
                                ,autocommit = True)

sql = """select distinct writetime
            from article 
            """
cursor.execute(sql)    
data = np.array(cursor.fetchall())


# In[ ]:


data


# In[ ]:




