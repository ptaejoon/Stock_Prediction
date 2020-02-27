import sys
from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
import time
import os
from datetime import datetime
import pickle
import numpy as np

class Kiwoom(QAxWidget):
    def __init__(self):
        super().__init__()
        self._create_kiwoom_instance()
        self._set_signal_slots()
        self.TR_REQ_TIME_INTERVAL = 0.2
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
        self.OnReceiveChejanData.connect(self._receive_chejan_data)
        self.OnReceiveRealData.connect(self._receive_real_data)
        self.OnReceiveRealCondition.connect(self._receive_real_condition)
        self.OnReceiveConditionVer.connect(self._receive_condition_ver)

    def comm_connect(self):
        self.dynamicCall("CommConnect()")
        self.login_event_loop = QEventLoop()
        self.login_event_loop.exec_()

    def get_chejan_data(self, fid):
        ret = self.dynamicCall("GetChejanData(int)", fid)
        return ret
        
    def _receive_chejan_data(self, gubun, item_cnt, fid_list):
        print(gubun)
        print(self.get_chejan_data(9203))
        print(self.get_chejan_data(302))
        print(self.get_chejan_data(900))
        print(self.get_chejan_data(901))
        self.order_event_loop.exit()

    def _receive_real_data(self, code, realtype, realdata):
        price = self._get_comm_real_data(code, 10)
        vol = self._get_comm_real_data(code, 13)
        print("code: ", code, "price: ", price, "volume: ", vol)

    def _receive_real_condition(self, code, event, condname, condind):
        print(code)
        print(event)
        print(condname)
        print(condind)
        self.real_event_loop.exit()

    def _receive_condition_ver(self, ret, msg):
        print(ret)
        print(msg)
        self.real_event_loop.exit()

    def _event_connect(self, err_code):
        if err_code == 0:
            print("connected")
        else:
            print("disconnected")

        self.login_event_loop.exit()

    def get_code_list_by_market(self, market):
        code_list = self.dynamicCall("GetCodeListByMarket(QString)", market)
        code_list = code_list.split(';')
        return code_list[:-1]

    def get_master_code_name(self, code):
        code_name = self.dynamicCall("GetMasterCodeName(QString)", code)
        return code_name

    def set_input_value(self, id, value):
        self.dynamicCall("SetInputValue(QString, QString)", id, value)

    def comm_rq_data(self, rqname, trcode, next, screen_no):
        self.dynamicCall("CommRqData(QString, QString, int, QString", rqname, trcode, next, screen_no)
        self.tr_event_loop = QEventLoop()
        self.tr_event_loop.exec_()

    def _get_comm_data(self, code, field_name, index, item_name):
        ret = self.dynamicCall("GetCommData(QString, QString, int, QString", code, field_name, index, item_name)
        return ret.strip()

    def _get_comm_real_data(self, code, fid):
        ret = self.dynamicCall("GetCommRealData(QString, int)", [code, fid])
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
        elif rqname == 'opt10001_req':
            self._opt10001(rqname, trcode)
        elif rqname == 'opt10003_req':
            self._opt10003(rqname, trcode)
        elif rqname == 'opt10004_req':
            self._opt10004(rqname, trcode)
        elif rqname == 'opt10005_req':
            self._opt10005(rqname, trcode)
        elif rqname == 'opt10007_req':
            self._opt10007(rqname, trcode)
        elif rqname == 'opt10073_req':
            self._opt10073(rqname, trcode)
        elif rqname == 'opt10075_req':
            self._opt10075(rqname, trcode)
        elif rqname == 'opt10076_req':
            self._opt10076(rqname, trcode)
        elif rqname == 'opt10077_req':
            self._opt10077(rqname, trcode)
        elif rqname == 'opt10084_req':
            self._opt10084(rqname, trcode)
        elif rqname == 'opt10086_req':
            self._opt10086(rqname, trcode)
        elif rqname == 'opw00004_req':
            self._opw00004(rqname, trcode)
        elif rqname == 'opw00007_req':
            self._opw00007(rqname, trcode)
            
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
            print(date, open, high, low, close, volume)
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

    def _opt10001(self, rqname, trcode):
        name = self._get_comm_data(trcode, rqname, 0, "종목명")
        price = self._get_comm_data(trcode, rqname, 0, "액면가")
        value = self._get_comm_data(trcode, rqname, 0, "시가총액")
        sell = self._get_comm_data(trcode, rqname, 0, "매출액")
        real_sell = self._get_comm_data(trcode, rqname, 0, "영업이익")
        start_price = self._get_comm_data(trcode, rqname, 0, "시가")
        high = self._get_comm_data(trcode, rqname, 0, "고가")
        low = self._get_comm_data(trcode, rqname, 0, "저가")
        print("종목명: ", name)
        print("액면가: ", price)
        print("시가총액: ", value)
        print("매출: " , sell)
        print("영업이익: ", real_sell)
        print("시가: ", start_price)
        print("고가: ", high)
        print("저가: ", low)

    def _opt10003(self, rqname, trcode):
        time = self._get_comm_data(trcode, rqname, 0, "시간")
        current_price = self._get_comm_data(trcode, rqname, 0, "현재가")
        prevday_ratio = self._get_comm_data(trcode, rqname, 0, "전일대비")
        ratio_percent = self._get_comm_data(trcode, rqname, 0, "대비율")
        unit = self._get_comm_data(trcode, rqname, 0, "우선매도호가단위")
        trade_volume = self._get_comm_data(trcode, rqname, 0, "체결거래량")
        sign = self._get_comm_data(trcode, rqname, 0, "sign")
        accum_volume = self._get_comm_data(trcode, rqname, 0, "누적거래량")
        power = self._get_comm_data(trcode, rqname, 0, "체결강도")

        print("시간 :", time)
        print("현재가 :", current_price)
        print("전일대비 :", prevday_ratio)
        print("대비율 :", ratio_percent)
        print("우선매도호가단위 :", unit)
        print("체결거래량 :", trade_volume)
        print("sign :", sign)
        print("누적거래량 :", accum_volume)
        print("체결강도 :", power)

    def _opt10004(self, rqname, trcode):
        sell_vol = [self._get_comm_data(trcode, rqname, 0, "매도최우선잔량")]
        buy_vol = [self._get_comm_data(trcode, rqname, 0, "매수최우선잔량")]
        sell_price = [self._get_comm_data(trcode, rqname, 0, "매도최우선호가")]
        buy_price = [self._get_comm_data(trcode, rqname, 0, "매수최우선호가")]
        sell_debi = []
        buy_debi = []
        
        for i in range(2, 6):
            sell_vol.append(self._get_comm_data(trcode, rqname, 0, "매도"+str(i)+"차선잔량"))
            sell_price.append(self._get_comm_data(trcode, rqname, 0, "매도"+str(i)+"차선호가"))
            sell_debi.append(self._get_comm_data(trcode, rqname, 0, "매도"+str(i)+"차선잔량대비"))
            buy_vol.append(self._get_comm_data(trcode, rqname, 0, "매수"+str(i)+"차선잔량"))
            buy_price.append(self._get_comm_data(trcode, rqname, 0, "매수"+str(i)+"차선호가"))
            buy_debi.append(self._get_comm_data(trcode, rqname, 0, "매수"+str(i)+"차선잔량대비"))

        time = self._get_comm_data(trcode, rqname, 0, "호가잔량기준시간")
        
        print("time: ", time)
        print("sell_vol: ", sell_vol)
        print("buy_vol", buy_vol)
        print("sell_price: ", sell_price)
        print("buy_price", buy_price)
        print("sell_debi: ", sell_debi)
        print("buy_debi", buy_debi)

    def _opt10005(self, rqname, trcode):
        time = self._get_comm_data(trcode, rqname, 0, "날짜")
        print(time)

    def _opt10007(self, rqname, trcode):
        time = self._get_comm_data(trcode, rqname, 0, "날짜")
        print(time)

    def _opt10073(self, rqname, trcode):
        date = self._get_comm_data(trcode, rqname, 0, "일자")
        print(date)

    def _opt10075(self, rqname, trcode):
        notpurchased_vol = self._get_comm_data(trcode, rqname, 0, "미체결수량")
        print(notpurchased_vol)

    def _opt10076(self, rqname, trcode):
        perchase_price = self._get_comm_data(trcode, rqname, 0, "체결가")
        print(perchase_price)

    def _opt10077(self, rqname, trcode):
        stock_name = self._get_comm_data(trcode, rqname, 0, "종목명")
        print(stock_name)

    def _opt10084(self, rqname, trcode):
        time = self._get_comm_data(trcode, rqname, 0, "시간")
        print(time)

    def _opt10086(self, rqname, trcode):
        data_cnt = self._get_repeat_cnt(trcode, rqname)
        for i in range(data_cnt):
            date = self._get_comm_data(trcode, rqname, i, "날짜")
            if datetime.strptime(date, "%Y%m%d") < self.start_date:
                self.remained_data = False
                return
            end_price = self._get_comm_data(trcode, rqname, i, "종가")
            start_price = self._get_comm_data(trcode, rqname, i, "시가")
            high_price = self._get_comm_data(trcode, rqname, i, "고가")
            low_price = self._get_comm_data(trcode, rqname, i, "저가")
            volume = self._get_comm_data(trcode, rqname, i, "거래량")
            print(date, end_price, start_price, high_price, low_price, volume)
            self.result.append([date, end_price, start_price, high_price, low_price, volume])

    def _opw00004(self, rqname, trcode):
        deposit = self._get_comm_data(trcode, rqname, 0, "예수금")
        print(deposit)
        month_profit = self._get_comm_data(trcode, rqname, 0, "누적손익율")
        print(month_profit)
        profit = self._get_comm_data(trcode, rqname, 0, "손익율")
        print(profit)

    def _opw00007(self, rqname, trcode):
        data_cnt = self._get_repeat_cnt(trcode, rqname)
        for i in range(data_cnt):
            time.sleep(self.TR_REQ_TIME_INTERVAL)
            num = self._get_comm_data(trcode, rqname, i, "주문번호")
            name = self._get_comm_data(trcode, rqname, i, "종목명")
            time = self._get_comm_data(trcode, rqname, i, "주문시간")
            print(num)
            print(name)
            print(time)

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
        while self.remained_data == True:
            time.sleep(self.TR_REQ_TIME_INTERVAL)
            self.set_input_value("종목코드", code)
            self.set_input_value("기준일자", date)
            self.set_input_value("수정주가구분", sujung)
            self.comm_rq_data("opt10081_req", "opt10081", 2, "0101")

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

    # [ opt10001 : 주식기본정보요청 ]
	# 종목코드 = 전문 조회할 종목코드
    def stock_basic_info_req(self, code):
        self.set_input_value("종목코드", code)
        self.comm_rq_data("opt10001_req", "opt10001", 0, "0106")

    # [ opt10003 : 체결정보정보요청 ]
	# 종목코드 = 전문 조회할 종목코드
    def sign_info_req(self, code):
        self.set_input_value("종목코드", code)
        self.comm_rq_data("opt10003_req", "opt10003", 0, "0107")
        
        while self.remained_data == True:
            time.sleep(self.TR_REQ_TIME_INTERVAL)
            self.set_input_value("send_order", code)
            self.comm_rq_data("opt10003_req", "opt10003", 2, "0107")

    def account_profit_req(self, acc_num):
        self.set_input_value("계좌번호", acc_num)
        self.comm_rq_data("opt10085_req", "opt10085", 0, "0108")

        while self.remained_data == True:
            time.sleep(self.TR_REQ_TIME_INTERVAL)
            self.set_input_value("계좌번호", acc_num)
            self.comm_rq_data("opt10085_req", "opt10085", 2, "0108")

    # [ 주식 매수, 취소, 정정 관련 함수]
    #  LONG OrderType,  // 주문유형 1:신규매수, 2:신규매도 3:매수취소, 4:매도취소, 5:매수정정, 6:매도정정
    #  Hoga
        #   00 : 지정가
        #   03 : 시장가
        #   05 : 조건부지정가
        #   06 : 최유리지정가
        #   07 : 최우선지정가
        #   10 : 지정가IOC
        #   13 : 시장가IOC
        #   16 : 최유리IOC
        #   20 : 지정가FOK
        #   23 : 시장가FOK
        #   26 : 최유리FOK
        #   61 : 장전시간외종가
        #   62 : 시간외단일가매매
        #   81 : 장후시간외종가
    def send_order(self, order_type, code, quantity, price, hoga, order_no):
        self.dynamicCall("SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",["send_order", "1010", "8130142611", order_type, code, quantity, price, hoga, order_no])
        self.order_event_loop = QEventLoop()
        self.order_event_loop.exec_()

    # [ opt10004 : 주식호가요청 ]
	# 종목코드 = 전문 조회할 종목코드
    def stock_price_level_req(self, code):
        self.set_input_value("종목코드", code)
        self.comm_rq_data("opt10004_req", "opt10004", 0, "0109")

    # [ opt10005 : 주식일주월시분요청  ]
	# 종목코드 = 전문 조회할 종목코드
    def stock_dayweekmonthtimemin_req(self, code):
        self.set_input_value("종목코드", code)
        self.comm_rq_data("opt10005_req", "opt10005", 0, "0111")

    # [ opt10007 : 시세표성정보요청  ]
	# 종목코드 = 전문 조회할 종목코드
    def price_current_info_req(self, code):
        self.set_input_value("종목코드", code)
        self.comm_rq_data("opt10007_req", "opt10007", 0, "0112")

    # [ opt10073 : 일자별종목별실현손익요청 ]
    # 	종목코드 = 전문 조회할 종목코드
    # 	시작일자 = YYYYMMDD (20160101 연도4자리, 월 2자리, 일 2자리 형식)
    # 	종료일자 = YYYYMMDD (20160101 연도4자리, 월 2자리, 일 2자리 형식)
    def perday_perstock_realprofit_req(self, code, start_date, end_date):
        self.set_input_value("계좌번호", "8130142611")
        self.set_input_value("종목코드", code)
        self.set_input_value("시작일자", start_date)
        self.set_input_value("종료일자", end_date)
        self.comm_rq_data("opt10085_req", "opt10085", 0, "0113")

    # [ opt10075 : 실시간미체결요청 ]
    # 	전체종목구분 = 0:전체, 1:종목
    # 	매매구분 = 0:전체, 1:매도, 2:매수
    # 	종목코드 = 전문 조회할 종목코드
    # 	체결구분 = 0:전체, 2:체결, 1:미체결
    def realtime_notperchased_req(self, stockall_flag, buysell_flag, code, contract_flag):
        self.set_input_value("계좌번호", "8130142611")
        self.set_input_value("전체종목구분", stockall_flag)
        self.set_input_value("매매구분", buysell_flag)
        self.set_input_value("종목코드", code)
        self.set_input_value("체결구분", contract_flag)
        self.comm_rq_data("opt10075_req", "opt10075", 0, "0114")

    #  [ opt10076 : 실시간체결요청 ]
    # 	종목코드 = 전문 조회할 종목코드
    # 	조회구분 = 0:전체, 1:종목
    # 	매도수구분 = 0:전체, 1:매도, 2:매수
    # 	주문번호 = 조회할 주문번호
    # 	체결구분 = 0:전체, 2:체결, 1:미체결
    def realtime_perchased_req(self, code, stockall_flag, buysell_flag, order_num, contract_flag):
        self.set_input_value("종목코드",  code)
        self.set_input_value("조회구분",  stockall_flag)
        self.set_input_value("매도수구분", buysell_flag)
        self.set_input_value("계좌번호", "8130142611")
        self.set_input_value("비밀번호", "")
        self.set_input_value("주문번호", order_num)
        self.set_input_value("체결구분", contract_flag)
        self.comm_rq_data("opt10076_req", "opt10076", 0, "0115")

    #  [ opt10077 : 당일실현손익상세요청 ]
    # 	종목코드 = 전문 조회할 종목코드
    def day_profitloss_req(self, code):
        self.set_input_value("계좌번호",  "8130142611")
        self.set_input_value("비밀번호",  "")
        self.set_input_value("종목코드",  code)
        self.comm_rq_data("opt10077_req", "opt10077", 0, "0116")
    
    #  [ opt10084 : 당일전일체결요청 ]
    # 	종목코드 = 전문 조회할 종목코드
    # 	당일전일 = 당일 : 1, 전일 : 2
    # 	틱분 = 틱 : 0 , 분 : 1
    # 	시간 = 조회시간 4자리, 오전 9시일 경우 '0900', 오후 2시 30분일 경우 '1430'
    def today_prevday_contract_req(self, code, today_prev, tick_min, time):
        self.set_input_value("종목코드",  code)
        self.set_input_value("당일전일",  today_prev)
        self.set_input_value("틱분",  tick_min)
        self.set_input_value("시간",  time)
        self.comm_rq_data("opt10084_req", "opt10084", 0, "0117")

    #  [ opt10086 : 일별주가요청 ]
    # 	종목코드 = 전문 조회할 종목코드
    # 	조회일자 = YYYYMMDD (20160101 연도4자리, 월 2자리, 일 2자리 형식)
    # 	표시구분 = 0:수량, 1:금액(백만원)
    def per_day_stockprice_req(self, code, date, present):
        self.result = []
        self.set_input_value("종목코드",  code)
        self.set_input_value("조회일자", date)
        self.set_input_value("표시구문",  present)
        self.comm_rq_data("opt10086_req", "opt10086", 0, "0118")

        while self.remained_data == True:
            time.sleep(self.TR_REQ_TIME_INTERVAL)
            self.set_input_value("종목코드",  code)
            self.set_input_value("조회일자", date)
            self.set_input_value("표시구문",  present)
            self.comm_rq_data("opt10086_req", "opt10086", 2, "0118")

#     [ OPW00004 : 계좌평가현황요청 ]
#     계좌번호 = 전문 조회할 보유계좌번호
#     비밀번호 = 사용안함(공백)
#     상장폐지조회구분 = 0:전체, 1:상장폐지종목제외
#     비밀번호입력매체구분 = 00
    def account_evaluation_req(self):
        self.set_input_value("계좌번호",  "8130142611")
        self.set_input_value("비밀번호",  "")
        self.set_input_value("상장폐지조회구분",  "0")
        self.set_input_value("비밀번호입력매체구분",  '00')
        self.comm_rq_data("opw00004_req", "opw00004", 0, "0119")

#  [ OPW00007 : 계좌별주문체결내역상세요청 ]
# 	주문일자 = YYYYMMDD (20170101 연도4자리, 월 2자리, 일 2자리 형식)
# 	조회구분 = 1:주문순, 2:역순, 3:미체결, 4:체결내역만
# 	매도수구분 = 0:전체, 1:매도, 2:매수
# 	종목코드 = 전문 조회할 종목코드
# 	시작주문번호 = 
    def account_orderdetail_req(self, date, inquiry, sell_buy, code, order_num):
        self.set_input_value("주문일자",  date)
        self.set_input_value("계좌번호",  "8130142611")
        self.set_input_value("비밀번호",  "")
        self.set_input_value("비밀번호입력매체구분",  "00")
        self.set_input_value("조회구분",  inquiry)
        self.set_input_value("주식채권구분",  "1")
        self.set_input_value("매도수구분",  sell_buy)
        self.set_input_value("종목코드",  code)
        self.set_input_value("시작주문번호",  "")
        self.comm_rq_data("opw00007_req", "opw00007", 0, "0120")

        while self.remained_data == True:
            time.sleep(self.TR_REQ_TIME_INTERVAL)
            self.set_input_value("주문일자",  date)
            self.set_input_value("계좌번호",  "8130142611")
            self.set_input_value("비밀번호",  "")
            self.set_input_value("비밀번호입력매체구분",  "00")
            self.set_input_value("조회구분",  inquiry)
            self.set_input_value("주식채권구분",  "1")
            self.set_input_value("매도수구분",  sell_buy)
            self.set_input_value("종목코드",  code)
            self.set_input_value("시작주문번호",  "")
            self.comm_rq_data("opw00007_req", "opw00007", 2, "0120")

    # [ 실시간 정보 받아오기 ]
    # strScreenNo = “0001” (화면번호)
    # strCodeList = “039490;005930;…” (종목코드 리스트)
    # strFidList = “9001;10;13;…” (FID 번호 리스트)
    # strOptType = “0” (타입)
    def get_real_data(self, codearr, fidarr, opt):
        self.dynamicCall("SetRealReg(QString, QString, QString, QString)", ["0121", codearr, fidarr, opt])
        self.real_event_loop = QEventLoop()
        self.real_event_loop.exec_()

    def remove_real_data(self, codearr):
        self.dynamicCall("SetRealRemove(QString, QString)", ["ALL", "ALL"])
        self.real_event_loop = QEventLoop()
        self.real_event_loop.exec_()

'''
실시간 데이터 조회의 경우 KOA에서 조건검색의 setRealReg를 참고할 것
'''
def load_predict_file():
    with open("predict.pickle", "rb") as f:
        predict = pickle.load(f)
        return predict.reshape([159, 4])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    kiwoom = Kiwoom()
    kiwoom.comm_connect()
    #kiwoom.day_stockdata_req("005930", "20200214", "256")
    # kiwoom.remove_real_data("005930")


    # [ 주식 매수, 취소, 정정 관련 함수]
    #  LONG OrderType,  // 주문유형 1:신규매수, 2:신규매도 3:매수취소, 4:매도취소, 5:매수정정, 6:매도정정
    #  Hoga
        #   00 : 지정가
        #   03 : 시장가
        #   05 : 조건부지정가
        #   06 : 최유리지정가
        #   07 : 최우선지정가
        #   10 : 지정가IOC
        #   13 : 시장가IOC
        #   16 : 최유리IOC
        #   20 : 지정가FOK
        #   23 : 시장가FOK
        #   26 : 최유리FOK
        #   61 : 장전시간외종가
        #   62 : 시간외단일가매매
        #   81 : 장후시간외종가