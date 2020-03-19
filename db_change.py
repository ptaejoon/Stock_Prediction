import pymysql
import trade_origin
import sys
from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
import time
import os
from datetime import datetime
import numpy as np

class Kiwoom(QAxWidget):
    def __init__(self):
        super().__init__()
        self._create_kiwoom_instance()
        self._set_signal_slots()
        self.TR_REQ_TIME_INTERVAL = 0.4
        self.start_date = datetime(2010, 1, 1)
        self.corp = [('006840', 'AK홀딩스'), ('001040', 'CJ'), ('079160', 'CJ CGV'), ('000120', 'CJ대한통운'), ('097950', 'CJ제일제당'),
             ('005830', 'DB손해보험'), ('000990', 'DB하이텍'), ('114090', 'GKL'), ('078930', 'GS'), ('006360', 'GS건설'),
             ('012630', 'HDC'), ('001060', 'JW중외제약'), ('096760', 'JW홀딩스'), ('105560', 'KB금융'), ('002380', 'KCC'),
             ('030200', 'KT'), ('033780', 'KT&G'), ('093050', 'LF'), ('003550', 'LG'), ('034220', 'LG디스플레이'),
             ('001120', 'LG상사'), ('051900', 'LG생활건강'), ('032640', 'LG유플러스'), ('011070', 'LG이노텍'), ('066570', 'LG전자'),
             ('108670', 'LG하우시스'), ('051910', 'LG화학'), ('006260', 'LS'), ('010120', 'LS산전'), ('035420', 'NAVER'),
             ('005940', 'NH투자증권'), ('010060', 'OCI'), ('005490', 'POSCO'), ('064960', 'S&T모티브'), ('010950', 'S-Oil'),
             ('034730', 'SK'), ('011790', 'SKC'), ('001740', 'SK네트웍스'), ('006120', 'SK디스커버리'), ('096770', 'SK이노베이션'),
             ('017670', 'SK텔레콤'), ('000660', 'SK하이닉스'), ('005610', 'SPC삼립'), ('035250', '강원랜드'), ('010130', '고려아연'),
             ('002240', '고려제강'), ('011780', '금호석유'), ('073240', '금호타이어'), ('000270', '기아차'), ('024110', '기업은행'),
             ('003920', '남양유업'), ('025860', '남해화학'), ('002350', '넥센타이어'), ('006280', '녹십자'), ('005250', '녹십자홀딩스'),
             ('004370', '농심'), ('019680', '대교'), ('008060', '대덕전자'), ('000210', '대림산업'), ('001680', '대상'),
             ('047040', '대우건설'), ('042660', '대우조선해양'), ('069620', '대웅제약'), ('006650', '대한유화'), ('003490', '대한항공'),
             ('001230', '동국제강'), ('026960', '동서'), ('000640', '동아쏘시오홀딩스'), ('001520', '동양'), ('049770', '동원F&B'),
             ('014820', '동원시스템즈'), ('000150', '두산'), ('042670', '두산인프라코어'), ('034020', '두산중공업'), ('023530', '롯데쇼핑'),
             ('004000', '롯데정밀화학'), ('004990', '롯데지주'), ('005300', '롯데칠성'), ('011170', '롯데케미칼'), ('002270', '롯데푸드'),
             ('008560', '메리츠종금증권'), ('006800', '미래에셋대우'), ('003850', '보령제약'), ('003000', '부광약품'), ('005180', '빙그레'),
             ('006400', '삼성SDI'), ('028050', '삼성엔지니어링'), ('009150', '삼성전기'), ('005930', '삼성전자'), ('010140', '삼성중공업'),
             ('016360', '삼성증권'), ('029780', '삼성카드'), ('000810', '삼성화재'), ('000070', '삼양홀딩스'), ('004490', '세방전지'),
             ('001430', '세아베스틸'), ('068270', '셀트리온'), ('004170', '신세계'), ('055550', '신한지주'), ('003410', '쌍용양회'),
             ('003620', '쌍용차'), ('002790', '아모레G'), ('090430', '아모레퍼시픽'), ('010780', '아이에스동서'), ('005850', '에스엘'),
             ('012750', '에스원'), ('036570', '엔씨소프트'), ('111770', '영원무역'), ('003520', '영진약품'), ('000670', '영풍'),
             ('007310', '오뚜기'), ('001800', '오리온홀딩스'), ('021240', '웅진코웨이'), ('014830', '유니드'), ('000100', '유한양행'),
             ('007570', '일양약품'), ('030000', '제일기획'), ('035720', '카카오'), ('003240', '태광산업'), ('028670', '팬오션'),
             ('047050', '포스코인터내셔널'), ('103140', '풍산'), ('086790', '하나금융지주'), ('000080', '하이트진로'), ('036460', '한국가스공사'),
             ('071050', '한국금융지주'), ('025540', '한국단자'), ('002960', '한국쉘석유'), ('015760', '한국전력'), ('009540', '한국조선해양'),
             ('000240', '한국테크놀로지그룹'), ('008930', '한미사이언스'), ('009240', '한샘'), ('020000', '한섬'), ('105630', '한세실업'),
             ('014680', '한솔케미칼'), ('018880', '한온시스템'), ('009420', '한올바이오파마'), ('006390', '한일현대시멘트'),
             ('051600', '한전KPS'), ('052690', '한전기술'), ('000880', '한화'), ('012450', '한화에어로스페이스'), ('009830', '한화케미칼'),
             ('000720', '현대건설'), ('005440', '현대그린푸드'), ('086280', '현대글로비스'), ('079430', '현대리바트'), ('012330', '현대모비스'),
             ('010620', '현대미포조선'), ('069960', '현대백화점'), ('017800', '현대엘리베이'), ('004020', '현대제철'), ('005380', '현대차'),
             ('001450', '현대해상'), ('008770', '호텔신라'), ('004800', '효성'), ('093370', '후성'), ('069260', '휴켐스')]

    def _create_kiwoom_instance(self):
        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")

    def _set_signal_slots(self):
        self.OnEventConnect.connect(self._event_connect)
        self.OnReceiveTrData.connect(self._receive_tr_data)

    def comm_connect(self):
        self.dynamicCall("CommConnect()")
        self.login_event_loop = QEventLoop()
        self.login_event_loop.exec_()

    def _event_connect(self, err_code):
        if err_code == 0:
            print("connected")
        else:
            print("disconnected")

        self.login_event_loop.exit()

    def set_input_value(self, id, value):
        self.dynamicCall("SetInputValue(QString, QString)", id, value)

    def comm_rq_data(self, rqname, trcode, next, screen_no):
        self.dynamicCall("CommRqData(QString, QString, int, QString", rqname, trcode, next, screen_no)
        self.tr_event_loop = QEventLoop()
        self.tr_event_loop.exec_()

    def _get_comm_data(self, code, field_name, index, item_name):
        ret = self.dynamicCall("GetCommData(QString, QString, int, QString", code, field_name, index, item_name)
        return ret.strip()

    def _get_repeat_cnt(self, trcode, rqname):
        ret = self.dynamicCall("GetRepeatCnt(QString, QString)", trcode, rqname)
        return ret

    # screen_no : 스크린 숫자
    # rqname : 사용자구분명
    # trcode : transaction code (ex. opt10001)
    # recode_name : 레코드 이름
    # next : 연속조회 유무를 판단하는 값 0: 연속(추가조회)데이터 없음, 2:연속(추가조회) 데이터 있음
    def _receive_tr_data(self, screen_no, rqname, trcode, record_name, next, unused1, unused2, unused3, unused4):
        if next == '2':
            self.remained_data = True
        else:
            self.remained_data = False

        if rqname == "opt10081_req":
            self._opt10081(rqname, trcode)
        elif rqname == 'opt10080_req':
            self._opt10080(rqname, trcode)
        elif rqname == 'opt10079_req':
            self._opt10079(rqname, trcode)
        elif rqname == 'opt10082_req':
            self._opt10082(rqname, trcode)
        elif rqname == 'opt10083_req':
            self._opt10083(rqname, trcode)
            
        try:
            self.tr_event_loop.exit()
        except AttributeError:
            pass

    def _opt10081(self, rqname, trcode):
        data_cnt = self._get_repeat_cnt(trcode, rqname)
        for i in range(data_cnt):
            date = self._get_comm_data(trcode, rqname, i, "일자")
            if self.result:
                if datetime.strptime(date, "%Y%m%d") < self.start_date:
                    self.remained_data = False
                    return
            open = self._get_comm_data(trcode, rqname, i, "시가")
            high = self._get_comm_data(trcode, rqname, i, "고가")
            low = self._get_comm_data(trcode, rqname, i, "저가")
            close = self._get_comm_data(trcode, rqname, i, "현재가")
            volume = self._get_comm_data(trcode, rqname, i, "거래량")
            #print(date, open, high, low, close, volume)
            self.result.append([date, open, high, low, close, volume])

    def _opt10080(self, rqname, trcode):
        data_cnt = self._get_repeat_cnt(trcode, rqname)

        for i in range(data_cnt):
            date = self._get_comm_data(trcode, rqname, i, "체결시간")
            open = self._get_comm_data(trcode, rqname, i, "시가")
            high = self._get_comm_data(trcode, rqname, i, "고가")
            low = self._get_comm_data(trcode, rqname, i, "저가")
            close = self._get_comm_data(trcode, rqname, i, "현재가")
            volume = self._get_comm_data(trcode, rqname, i, "거래량")
            print(date, open, high, low, close, volume)
    
    def _opt10079(self, rqname, trcode):
        data_cnt = self._get_repeat_cnt(trcode, rqname)

        for i in range(data_cnt):
            date = self._get_comm_data(trcode, rqname, i, "체결시간")
            open = self._get_comm_data(trcode, rqname, i, "시가")
            high = self._get_comm_data(trcode, rqname, i, "고가")
            low = self._get_comm_data(trcode, rqname, i, "저가")
            close = self._get_comm_data(trcode, rqname, i, "현재가")
            volume = self._get_comm_data(trcode, rqname, i, "거래량")
            print(date, open, high, low, close, volume)

    def _opt10082(self, rqname, trcode):
        data_cnt = self._get_repeat_cnt(trcode, rqname)

        for i in range(data_cnt):
            date = self._get_comm_data(trcode, rqname, i, "일자")
            open = self._get_comm_data(trcode, rqname, i, "시가")
            high = self._get_comm_data(trcode, rqname, i, "고가")
            low = self._get_comm_data(trcode, rqname, i, "저가")
            close = self._get_comm_data(trcode, rqname, i, "현재가")
            volume = self._get_comm_data(trcode, rqname, i, "거래량")
            print(date, open, high, low, close, volume)

    def _opt10083(self, rqname, trcode):
        data_cnt = self._get_repeat_cnt(trcode, rqname)

        for i in range(data_cnt):
            date = self._get_comm_data(trcode, rqname, i, "일자")
            open = self._get_comm_data(trcode, rqname, i, "시가")
            high = self._get_comm_data(trcode, rqname, i, "고가")
            low = self._get_comm_data(trcode, rqname, i, "저가")
            close = self._get_comm_data(trcode, rqname, i, "현재가")
            volume = self._get_comm_data(trcode, rqname, i, "거래량")
            print(date, open, high, low, close, volume)

    # [ opt10081 : 주식일봉차트조회요청 ]
    #     종목코드 = 전문 조회할 종목코드
    #     기준일자 = YYYYMMDD (20160101 연도4자리, 월 2자리, 일 2자리 형식)
    #     수정주가구분 = 0 or 1, 수신데이터 1:유상증자, 2:무상증자, 4:배당락, 8:액면분할, 16:액면병합, 32:기업합병, 64:감자, 256:권리락
    def day_stockdata_req(self, code, date, sujung):
        self.result = []
        self.set_input_value("종목코드", code)
        self.set_input_value("기준일자", date)
        self.set_input_value("수정주가구분", sujung)
        self.comm_rq_data("opt10081_req", "opt10081", 0, "0101")

    # [ opt10080 : 주식분봉차트조회요청 ]
    #     종목코드 = 전문 조회할 종목코드
    #     틱범위 = 1:1분, 3:3분, 5:5분, 10:10분, 15:15분, 30:30분, 45:45분, 60:60분
    #     수정주가구분 = 0 or 1, 수신데이터 1:유상증자, 2:무상증자, 4:배당락, 8:액면분할, 16:액면병합, 32:기업합병, 64:감자, 256:권리락
    def min_stockdata_req(self, code, minute, sujung):
        self.set_input_value("종목코드", code)
        self.set_input_value("틱범위", minute)
        self.set_input_value("수정주가구분", sujung)
        self.comm_rq_data("opt10080_req", "opt10080", 0, "0102")

        while self.remained_data == True:
            time.sleep(self.TR_REQ_TIME_INTERVAL)
            self.set_input_value("종목코드", code)
            self.set_input_value("틱범위", minute)
            self.set_input_value("수정주가구분", sujung)
            self.comm_rq_data("opt10080_req", "opt10080", 2, "0102")

    #  [ opt10079 : 주식틱차트조회요청 ]
    #     종목코드 = 전문 조회할 종목코드
    #     틱범위 = 1:1틱, 3:3틱, 5:5틱, 10:10틱, 30:30틱
    #     수정주가구분 = 0 or 1, 수신데이터 1:유상증자, 2:무상증자, 4:배당락, 8:액면분할, 16:액면병합, 32:기업합병, 64:감자, 256:권리락
    def tick_stockdata_req(self, code, tick, sujung):
        self.set_input_value("종목코드", code)
        self.set_input_value("틱범위", tick)
        self.set_input_value("수정주가구분", sujung)
        self.comm_rq_data("opt10079_req", "opt10079", 0, "0103")

        while self.remained_data == True:
            time.sleep(self.TR_REQ_TIME_INTERVAL)
            self.set_input_value("종목코드", code)
            self.set_input_value("틱범위", tick)
            self.set_input_value("수정주가구분", sujung)
            self.comm_rq_data("opt10079_req", "opt10079", 2, "0103")
    
    # [ opt10082 : 주식주봉차트조회요청 ]
    # 	종목코드 = 전문 조회할 종목코드
    # 	기준일자 = YYYYMMDD (20160101 연도4자리, 월 2자리, 일 2자리 형식)
    # 	수정주가구분 = 0 or 1, 수신데이터 1:유상증자, 2:무상증자, 4:배당락, 8:액면분할, 16:액면병합, 32:기업합병, 64:감자, 256:권리락
    def week_stockdata_req(self, code, date, lastdate,sujung):
        self.set_input_value("종목코드", code)
        self.set_input_value("기준일자", date)
        self.set_input_value("끝일자", lastdate)
        self.set_input_value("수정주가구분", sujung)
        self.comm_rq_data("opt10082_req", "opt10082", 0, "0104")

        # while self.remained_data == True:
        #     self.set_input_value("종목코드", code)
        #     self.set_input_value("기준일자", code)
        #     self.set_input_value("끝일자", code)
        #     self.set_input_value("수정주가구분", code)
        #     self.comm_rq_data("opt10082_req", "opt10082", 2, "0104")   

    #  [ opt10083 : 주식월봉차트조회요청 ]
    #     종목코드 = 전문 조회할 종목코드
    #     기준일자 = YYYYMMDD (20160101 연도4자리, 월 2자리, 일 2자리 형식)
    #     수정주가구분 = 0 or 1, 수신데이터 1:유상증자, 2:무상증자, 4:배당락, 8:액면분할, 16:액면병합, 32:기업합병, 64:감자, 256:권리락
    def month_stockdata_req(self, code, date, lastdate, sujung):
        self.set_input_value("종목코드", code)
        self.set_input_value("기준일자", date)
        self.set_input_value("끝일자", lastdate)
        self.set_input_value("수정주가구분", sujung)
        self.comm_rq_data("opt10083_req", "opt10083", 0, "0105")

        while self.remained_data == True:
            time.sleep(self.TR_REQ_TIME_INTERVAL)
            self.set_input_value("종목코드", code)
            self.set_input_value("기준일자", date)
            self.set_input_value("끝일자", lastdate)
            self.set_input_value("수정주가구분", sujung)
            self.comm_rq_data("opt10083_req", "opt10083", 2, "0105")

if __name__ == "__main__":
    if datetime.today().weekday()== 5 or datetime.today().weekday()== 6:
        exit()
    conn = pymysql.connect(host='sp-articledb.clwrfz92pdul.ap-northeast-2.rds.amazonaws.com', user = 'admin', password='sogangsp', db='mydb', charset='utf8', port=3306)
    curs = conn.cursor()
    app = QApplication(sys.argv)
    kiwoom = Kiwoom()
    kiwoom.comm_connect()
    print("오늘 날짜", datetime.today())
    for row in kiwoom.corp:
        kiwoom.day_stockdata_req(row[0], datetime.today().strftime("%Y%m%d"), "256")
        for day_data in kiwoom.result:
            sql = "insert into stock(trade_time, corp_name, open_price, max_price, min_price, close_price, trade_amount) values('"+day_data[0]+"', '"+row[1]+"', "+day_data[1]+", "+day_data[2]+", "+day_data[3]+", "+day_data[4]+", "+day_data[5]+");"
            curs.execute(sql)
            break
        conn.commit()
        print(row[1])
        time.sleep(2)