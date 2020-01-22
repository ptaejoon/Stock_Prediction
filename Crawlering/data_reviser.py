import pymysql
import datetime
monthly_day = [31,28,31,30,31,30,31,31,30,31,30,31]
monthly_day_leap = [31,29,31,30,31,30,31,31,30,31,30,31]

processedDB = {"host" : 'sp-articledb.clwrfz92pdul.ap-northeast-2.rds.amazonaws.com',"port":3306,"user":'admin',"password":"sogangsp","db":"mydb",'charset':'utf8'}
localDB = {'host':'127.0.0.1','port':3306,'user':'root','password':'sogangsp','db':'mydb','charset':'utf8'}

localDBconnect = pymysql.connect(
    host=localDB["host"],
    port =localDB['port'],
    user=localDB['user'],
    password=localDB['password'],
    db=localDB['db'],
    charset=localDB['charset']
                               )

#proCursor = proDBconnect.cursor()
#rawCursor = rawDBconnect.cursor()
localCursor = localDBconnect.cursor()


revise_sql = """UPDATE article SET writetime = concat(date(writetime),' 00:',minute(writetime),':',second(writetime))
where writetime >= concat(%s," 12:00:00") and writetime < concat(%s, " 13:00:00")
and newsid > (select newsid from( select newsid from article as a
where writetime > concat(%s, " 01:00:00.000") and writetime < concat(%s, " 02:00:00.000") 
limit 1 )a);
"""

def revise12Time(even = False):
    evenYear = [2010,2012,2014,2016,2018]
    oddYear = [2011,2013,2015,2017,2019]
    if even is True:
        for year in oddYear:
            if year % 4 == 0:
                present_month = 1
                for month in monthly_day_leap:
                    for day in len(1,month+1):
                        year_string = str(year) + '-' + str(present_month) + '-' + str(day)
                        print(year_string)
                    present_month = present_month + 1
            else:
                present_month = 1
                for month in monthly_day:
                    for day in range(1,month+1):
                        year_string = str(year) + '-' + str(present_month) + '-' + str(day)
                        print(year_string)
                    present_month = present_month + 1
    else:
        for year in oddYear:
            present_month = 1
            for month in monthly_day:
                for day in range(1,month+1):
                    year_string = str(year)+'-'+str(present_month)+'-'+str(day)
                    reviseDB(year_string)
                present_month = present_month+1

def reviseDB(date):
    #date = datetime.datetime.strptime(date,'%Y-%m-%d').date()
    #print(type(date))
    args = (date,date,date,date)

    localCursor.execute(revise_sql,args)
    # rawCursor.execute(insert_raw_sql,(content,time,section))
    # proDBconnect.commit()
    # rawDBconnect.commit()
    localDBconnect.commit()
    print(date)

revise12Time()
