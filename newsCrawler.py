import requests
from bs4 import BeautifulSoup
import re
naver_news_link = "https://news.naver.com/main/list.nhn?mode=LSD&mid=sec"
section = ["100","101","102","103","104","105"]


def readOneNews(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'html.parser')
    except:
        print("page "+url+" failed to read")
        return
    try:
        titleform = soup.find('h3',id="articleTitle")
        title = titleform.get_text()
        timeform = soup.find('span',{'class':'t11'})
        time = timeform.get_text()
        contentform = soup.find('div',id="articleBodyContents")
        for ad in contentform.find_all('a'):
            ad.decompose()
        content = contentform.get_text()
        content = content.replace("// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}", "")
        content = content.replace("\n"," ")
        content = content.replace("\t"," ")
        print(title,time,content)
        return [title,time,content]
    except:
        print("기사 원문을 따오는데 실패하였습니다.")
        return

def readOneNewsList(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.content,'html.parser')
    except:
        print("page "+url+" failed to read")
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
                readOneNews(newsLink)
                prenewsLink = newsLink
                articleList.append(newsLink)
    except:
        print("한 페이지의 기사 리스트에서 기사 접근에 실패하였습니다.")
    return articleList

def readOneDayList(url):
    #url 예시 https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=104&date=20200106&page=
    pageNum = 1
    try:
        prevdayList = []
        dayList = readOneNewsList(url+str(pageNum))
        while prevdayList != dayList:
            pageNum = pageNum + 1
            prevdayList = dayList
            dayList = readOneNewsList(url+str(pageNum))
            print("page : " + str(pageNum))
    except:
        print("하루치 기사 크롤링에 실패하였습니다.")
        print("실패한 페이지 : " + str(pageNum))



readOneNews('https://news.naver.com/main/read.nhn?mode=LSD&mid=sec&sid1=100&oid=056&aid=0010780168')
readOneNewsList('https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=104&date=20200106')
readOneDayList('https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=100&date=20200107&page=')