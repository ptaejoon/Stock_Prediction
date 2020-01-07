import requests
from bs4 import BeautifulSoup
import re
naver_news_link = "https://news.naver.com/main/list.nhn?mode=LSD&mid=sec"
section = ["100","101","102","103","104","105"]

def readOneNews(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'html.parser')
    except :
        print("page "+url+" failed to read")
        return
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
    return [title,time,content]

readOneNews('https://news.naver.com/main/read.nhn?mode=LSD&mid=sec&sid1=100&oid=056&aid=0010780168')
readOneNews('https://news.naver.com/main/read.nhn?mode=LSD&mid=sec&sid1=105&oid=055&aid=0000174803')
readOneNews('https://news.naver.com/main/read.nhn?mode=LSD&mid=sec&sid1=104&oid=001&aid=0011320304')