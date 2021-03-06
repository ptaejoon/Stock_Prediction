#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
#from tokenizer import tokenize
import pymysql
import sys
from datetime import datetime,timedelta
monthly_day = [31,28,31,30,31,30,31,31,30,31,30,31]
monthly_day_leap = [31,29,31,30,31,30,31,31,30,31,30,31]

processedDB = {"host" : 'sp-articledb.clwrfz92pdul.ap-northeast-2.rds.amazonaws.com',"port":3306,"user":'admin',"password":"sogangsp","db":"mydb",'charset':'utf8'}
rawDB = {"host" : 'article-raw-data.cnseysfqrlcj.ap-northeast-2.rds.amazonaws.com',"port":3306,"user":'admin',"password":"sogangsp","db":"mydb",'charset':'utf8'}

proDBconnect = pymysql.connect(
    host=processedDB["host"],
    port =processedDB['port'],
    user=processedDB['user'],
    password=processedDB['password'],
    db=processedDB['db'],
    charset=processedDB['charset']
                               )

"""rawDBconnect =pymysql.connect(
    host=rawDB["host"],
    port=rawDB['port'],
    user=rawDB['user'],
    password=rawDB['password'],
    db=rawDB['db'],
    charset=rawDB['charset']
)"""
proCursor = proDBconnect.cursor()
#rawCursor = rawDBconnect.cursor()

def save_to_DB(time,content,section):
    try:
        insert_pro_sql = """INSERT INTO article(news, writetime,section) VALUES ( %s, %s, %s )"""
        #insert_raw_sql = """INSERT INTO article(news, writetime,section) VALUES ( %s, %s, %s )"""
        #print(token)
        #print(time)
        #print(section)
        proCursor.execute(insert_pro_sql,(content,time,section))
        #rawCursor.execute(insert_raw_sql,(content,time,section))
        proDBconnect.commit()
        #rawDBconnect.commit()
        #print("i worked")
    except Exception as e:
        print(e)

def timeChange(time):
    afternoon = 0
    if "오후" in time:
        afternoon = 12
    timeSplit = time.split('.')
    returntime = timeSplit[0]+'-' + timeSplit[1]+'-' + timeSplit[2]+' '
    hour_minute = timeSplit[-1].split(' ')[-1]
    hour = int(hour_minute.split(':')[0]) + afternoon
    if hour == 24 and "오후" in time:
        hour = 12
    minute = int(hour_minute.split(':')[1])
    returntime = returntime + str(hour)+':'+str(minute)
    returntime = datetime.strptime(returntime,'%Y-%m-%d %H:%M')
    #print(type(returntime))
    return returntime

def readOneNews(url,section,day):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'html.parser')
    except Exception as e:
        print("page "+url+" failed to read")
        print(e)
        return
    try:
        titleform = soup.find('h3',id="articleTitle")
        title = titleform.get_text()
        timeform = soup.find('span',{'class':'t11'})
        time = timeChange(timeform.get_text())
        t_year = time.year
        t_month = time.month
        t_day = time.day

        today = datetime(t_year, t_month, t_day, 9, 0, 0, 0) #기사 작성날의 9시
        yesterday = today - timedelta(days=1) #기사 작성날의 전날 9시
        if day is "today":
            if time < today:
                print("Crawl")
            else:
                print("수집하려는 시간과 맞지 않습니다.")
                return
        else :
            if time > yesterday:
                print("Crawl")
            else:
                print("수집하려는 시간과 맞지 않습니다.")
                return
        contentform = soup.find('div',id="articleBodyContents")
        for ad in contentform.find_all('a'):
            ad.decompose()
        content = contentform.get_text()
        content = title + content
        content = content.replace("// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}", "")
        content = content.replace("\n"," ")
        content = content.replace("\t"," ")
        #print(title,time,content)
        #contentToken = tokenize(content)
        #print(contentToken)
       
        save_to_DB(time,content,section)

    except KeyboardInterrupt as e:
        sys.exit()
    except Exception as e:
        print("기사 원문을 따오는데 실패하였습니다.")
        print(e)
        return

def readOneNewsList(url,section,day):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.content,'html.parser')
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        print("page "+url+" failed to read")
        print(e)
        return
    try:
        articleList = []
        postListform = soup.find_all('a',{'class':'nclicks(fls.list)'})
        prenewsLink = ''
        for postList in postListform:
            newsLink = postList.get('href')
            if prenewsLink == newsLink:
                prenewsLink = newsLink
                continue
            else:
                readOneNews(newsLink,section,day)
                prenewsLink = newsLink
                articleList.append(newsLink)
    except:
        print("한 페이지의 기사 리스트에서 기사 접근에 실패하였습니다.")
    return articleList

def readOneDayList(url,section,day):
    #url 예시 https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=104&date=20200106&page=
    pageNum = 1
    try:
        prevdayList = []
        dayList = readOneNewsList(url+str(pageNum),section,day)
        while prevdayList != dayList:
            print("page : " + str(pageNum))
            pageNum = pageNum + 1
            prevdayList = dayList
            dayList = readOneNewsList(url+str(pageNum),section,day)
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        print("하루치 기사 크롤링에 실패하였습니다.")
        print("실패한 페이지 : " + str(pageNum))
        print(e)
    
today = datetime.now()
yesterday = today-timedelta(days=1)
if today.month < 10:
    month_string = str('0' + str(today.month))
else:
    month_string = str(today.month)
today_link = str(today.year)+str(month_string)

if yesterday.month < 10:
    month_string = str('0' + str(yesterday.month))
else:
    month_string = str(yesterday.month)
yesterday_link = str(yesterday.year)+str(month_string)

if today.day < 10:
    day_string = str('0' + str(today.day))
else:
    day_string = str(today.day)
today_link = today_link+str(day_string)

if yesterday.day < 10:
    day_string = str('0' + str(yesterday.day))
else:
    day_string = str(yesterday.day)
yesterday_link = yesterday_link+str(day_string)
print(today_link)
print(yesterday_link)
readOneDayList('https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=100&date='+today_link+'&page=',"경제",'today')
readOneDayList('https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=100&date='+yesterday_link+'&page=',"경제",'yesterday')





