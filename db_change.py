import pymysql
import trade_origin
import sys
from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
import time
import os
import datetime


conn = pymysql.connect(host='sp-articledb.clwrfz92pdul.ap-northeast-2.rds.amazonaws.com', user = 'admin', password='sogangsp', db='mydb', charset='utf8', port=3306)
curs = conn.cursor()

app = trade_origin.QApplication(sys.argv)
kiwoom = trade_origin.Kiwoom()
kiwoom.comm_connect()

for row in kiwoom.corp[14:15]:
    kiwoom.day_stockdata_req(row[0], '20200311', "256")
    for day_data in kiwoom.result:
        # print(day_data)
        if datetime.datetime.strptime(day_data[0], "%Y%m%d") < datetime.datetime(2020, 3, 11):
            break
        sql = "insert into stock(trade_time, corp_name, open_price, max_price, min_price, close_price, trade_amount) values('"+day_data[0]+"', '"+row[1]+"', "+day_data[1]+", "+day_data[2]+", "+day_data[3]+", "+day_data[4]+", "+day_data[5]+");"
        curs.execute(sql)
    conn.commit()
    print(row[1])
    time.sleep(2)