import sys
from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
import time
import os

TR_REQ_TIME_INTERVAL = 0.2

class Kiwoom(QAxWidget):
    def __init__(self):
        super().__init__()
        self._create_kiwoom_instance()
        self._set_signal_slots()

    def _create_kiwoom_instance(self):
        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")

    def _set_signal_slots(self):
        self.OnEventConnect.connect(self._event_connect)
        self.OnReceiveTrData.connect(self._receive_tr_data)
        self.OnReceiveChejanData.connect(self._receive_chejan_data)

    def comm_connect(self):
        self.dynamicCall("CommConnect()")
        self.login_event_loop = QEventLoop()
        self.login_event_loop.exec_()

    def get_chejan_data(self, fid):
        ret = self.dynamicCall("GetChejanData(int)", fid)
        return ret
        
    def _receive_chejan_data(self, gubun, item_cnt, fid_list):
        print(gubun)
        print("Hi")
        print(self.get_chejan_data(9203))
        print(self.get_chejan_data(302))
        print(self.get_chejan_data(900))
        print(self.get_chejan_data(901))

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
        ret = self.dynamicCall("GetCommData(QString, QString, int, QString", code,
                               field_name, index, item_name)
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
            open = self._get_comm_data(trcode, rqname, i, "시가")
            high = self._get_comm_data(trcode, rqname, i, "고가")
            low = self._get_comm_data(trcode, rqname, i, "저가")
            close = self._get_comm_data(trcode, rqname, i, "현재가")
            volume = self._get_comm_data(trcode, rqname, i, "거래량")
            print(date, open, high, low, close, volume)

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
        price = self._get_comm_data(trcode, rqname, 0, "시가")
        print(price)

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
        self.set_input_value("종목코드", code)
        self.set_input_value("기준일자", date)
        self.set_input_value("수정주가구분", sujung)
        self.comm_rq_data("opt10081_req", "opt10081", 0, "0101")
        return
        while self.remained_data == True:
            time.sleep(TR_REQ_TIME_INTERVAL)
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
            time.sleep(TR_REQ_TIME_INTERVAL)
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
            time.sleep(TR_REQ_TIME_INTERVAL)
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
            time.sleep(TR_REQ_TIME_INTERVAL)
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
            time.sleep(TR_REQ_TIME_INTERVAL)
            self.set_input_value("send_order", code)
            self.comm_rq_data("opt10003_req", "opt10003", 2, "0107")

    def account_profit_req(self, acc_num):
        self.set_input_value("계좌번호", acc_num)
        self.comm_rq_data("opt10085_req", "opt10085", 0, "0108")

        while self.remained_data == True:
            time.sleep(TR_REQ_TIME_INTERVAL)
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
        self.tr_event_loop = QEventLoop()
        self.tr_event_loop.exec_()

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
        self.set_input_value("종목코드",  code)
        self.set_input_value("조회일자", date)
        self.set_input_value("표시구문",  present)
        self.comm_rq_data("opt10086_req", "opt10086", 0, "0118")

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
            time.sleep(TR_REQ_TIME_INTERVAL)
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

if __name__ == "__main__":
    #update_api()
    app = QApplication(sys.argv)
    kiwoom = Kiwoom()
    kiwoom.comm_connect()
    kiwoom.day_stockdata_req("271560", '20200217', "256")