import stockGAN
import datetime
import numpy as np

import pandas as pd
from sqlalchemy import create_engine
import pymysql 
pymysql.install_as_MySQLdb()
import MySQLdb
import os

def predict(days):
    #days : datetime
    gan = stockGAN.GAN(batch_size=40)
    gan.load(7400)
    #gan.train(1)
    #gan.load(num=0)
    days = str(days)
    time = ' 15:30:00'
    days_to_datetime = datetime.datetime.strptime(str(days+time),'%Y-%m-%d %H:%M:%S')
    return gan.predict(days_to_datetime)


def save_rds(data):
    engine = create_engine(
        "mysql+mysqldb://admin:"+"sogangsp"+"@sp-articledb.clwrfz92pdul.ap-northeast-2.rds.amazonaws.com/mydb"
        ,encoding = 'utf-8')
    conn = engine.connect()
    data.to_sql(name = 'predict_stock', con = engine, if_exists ='append',index = False)
    print("Saving SQL is completed")
    engine.dispose()
    conn.close()



def load_rds(s) :
    
    db = pymysql.connect(
        host = 'sp-articledb.clwrfz92pdul.ap-northeast-2.rds.amazonaws.com',
        port =  3306,
        user = 'admin',
        passwd = 'sogangsp',
        db = 'mydb',
        charset = 'utf8')
    cursor = db.cursor()
    sql = """
          select close_price from stock where trade_time
          ='{}';
          """.format(s)
    cursor.execute(sql)
    data = pd.DataFrame(cursor.fetchall() , columns = ['close_price'])
    db.close()

    return data



corp_list = [('006840', 'AK홀딩스'), ('001040', 'CJ'), ('079160', 'CJ CGV'), ('000120','CJ대한통운'),('097950', 'CJ제일제당'),
('005830', 'DB손해보험'), ('000990', 'DB하이텍'), ('114090', 'GKL'), ('078930','GS'), ('006360', 'GS건설'),('012630', 'HDC'), ('001060', 'JW중외제약'), ('096760', 'JW홀딩스'),
('105560', 'KB금융'), ('002380', 'KCC'),('030200', 'KT'), ('033780', 'KT&G'), ('093050', 'LF'), ('003550', 'LG'),
('034220', 'LG디스플레이'),('001120', 'LG상사'), ('051900', 'LG생활건강'), ('032640', 'LG유플러스'),('011070', 'LG이노텍'), ('066570', 'LG전자'),
('108670', 'LG하우시스'), ('051910', 'LG화학'), ('006260', 'LS'),('010120', 'LS산전'), ('035420', 'NAVER'), ('005940', 'NH투자증권'), ('010060', 'OCI'), ('005490', 'POSCO'),
('064960', 'S&T모티브'), ('010950', 'S-Oil'),('034730', 'SK'), ('011790', 'SKC'), ('001740', 'SK네트웍스'), ('006120','SK디스커버리'), ('096770', 'SK이노베이션'),
('017670', 'SK텔레콤'), ('000660', 'SK하이닉스'), ('005610', 'SPC삼립'),('035250', '강원랜드'), ('010130', '고려아연'),
('002240', '고려제강'), ('011780', '금호석유'), ('073240','금호타이어'),('000270', '기아차'), ('024110', '기업은행'),('003920', '남양유업'),
('025860', '남해화학'), ('002350','넥센타이어'),('006280', '녹십자'), ('005250', '녹십자홀딩스'),('004370', '농심'), ('019680', '대교'), ('008060', '대덕전자'),('000210', '대림산업'), ('001680', '대상'),
('047040', '대우건설'), ('042660', '대우조선해양'), ('069620','대웅제약'), ('006650', '대한유화'), ('003490', '대한항공'),('001230', '동국제강'), ('026960', '동서'), ('000640','동아쏘시오홀딩스'),
('001520', '동양'), ('049770', '동원F&B'),('014820', '동원시스템즈'), ('000150', '두산'), ('042670','두산인프라코어'), ('034020', '두산중공업'), ('023530','롯데쇼핑'),
('004000', '롯데정밀화학'), ('004990', '롯데지주'),('005300', '롯데칠성'), ('011170', '롯데케미칼'), ('002270','롯데푸드'),('008560', '메리츠종금증권'), ('006800', '미래에셋대우'),
('003850', '보령제약'), ('003000', '부광약품'), ('005180','빙그레'),('006400', '삼성SDI'), ('028050', '삼성엔지니어링'),('009150', '삼성전기'), ('005930', '삼성전자'), ('010140','삼성중공업'),
('016360', '삼성증권'), ('029780', '삼성카드'), ('000810','삼성화재'), ('000070', '삼양홀딩스'), ('004490','세방전지'),('001430', '세아베스틸'), ('068270', '셀트리온'),
('004170', '신세계'), ('055550', '신한지주'),('003410', '쌍용양회'),('003620', '쌍용차'), ('002790', '아모레G'),('090430', '아모레퍼시픽'), ('010780','아이에스동서'), ('005850', '에스엘'),
('012750', '에스원'), ('036570', '엔씨소프트'),('111770', '영원무역'), ('003520', '영진약품'),('000670', '영풍'),('007310', '오뚜기'), ('001800', '오리온홀딩스'), ('021240', '웅진코웨이'),
('014830', '유니드'), ('000100', '유한양행'),('007570', '일양약품'), ('030000', '제일기획'), ('035720', '카카오'),('003240', '태광산업'), ('028670', '팬오션'),
('047050', '포스코인터내셔널'), ('103140', '풍산'), ('086790','하나금융지주'), ('000080', '하이트진로'), ('036460','한국가스공사'),('071050', '한국금융지주'), ('025540', '한국단자'),
('002960','한국쉘석유'), ('015760', '한국전력'), ('009540','한국조선해양'),('000240', '한국테크놀로지그룹'), ('008930','한미사이언스'), ('009240', '한샘'), ('020000','한섬'), ('105630', '한세실업'),
('014680', '한솔케미칼'), ('018880','한온시스템'), ('009420','한올바이오파마'), ('006390','한일현대시멘트'),('051600', '한전KPS'), ('052690','한전기술'), ('000880','한화'),
('012450','한화에어로스페이스'),('009830', '한화케미칼'),('000720', '현대건설'),('005440', '현대그린푸드'),('086280','현대글로비스'),('079430','현대리바트'),('012330','현대모비스'),('010620','현대미포조선'),
('069960','현대백화점'),('017800','현대엘리베이'),('004020', '현대제철'), ('005380', '현대차'), ('001450','현대해상'), ('008770','호텔신라'),('004800', '효성'), ('093370', '후성'),('069260', '휴켐스')]




if __name__ == '__main__':
    #days : datetime
    os.chdir("Desktop/udf/")

    # 예측일  + timedelta
    now = datetime.datetime.now()#-datetime.timedelta(days=1)
    day = now.date() + datetime.timedelta(days=1) # - -> +
    day_before = now.date() #- datetime.timedelta(days=1) #str

    # 종목코드 / 기업명
    corp_data = pd.DataFrame(corp_list, columns = ["code","corp_name"])
    # 예측 value Close/Open/Max/Min
    predict_np_data = predict(day).reshape(159,4)
    predict_df_data = pd.DataFrame(predict_np_data,columns = ["close_price","open_price","max_price","min_price"])
    print(now)
    print(day)
    print(day_before)
    #print(predict_np_data)
    # 종가곱하기
    if now.weekday() == 4 :
        day = day + datetime.timedelta(days=2)
    #if now.weekday() == 0 :
    #    day_before = day_before - datetime.timedelta(days=3)
    yesterday_close = load_rds(str(day_before))

    while True :
        if yesterday_close["close_price"].isnull().sum() >= 1 :
            yesterday_close = load_rds(str(day_before - datetime.timedelta(days=1)))
        else :
            break
    #print(yesterday_close)
    predict_df_data["close_price"] = predict_df_data["close_price"] * yesterday_close["close_price"]
    predict_df_data["open_price"] = predict_df_data["open_price"] * yesterday_close["close_price"]
    predict_df_data["max_price"] = predict_df_data["max_price"] * yesterday_close["close_price"]
    predict_df_data["min_price"] = predict_df_data["min_price"] * yesterday_close["close_price"]

    # 예측날짜 DataFrame 생성
    date_df_data = pd.DataFrame(index = range(0,159), columns = ['trade_time'])
    date_df_data[date_df_data.isna()] = day
            
    # predict dataframe to RDS for using sql 
    to_sql_data = pd.concat([date_df_data,corp_data,predict_df_data],axis=1)
    
    print(to_sql_data) # 최종 예측/ RDS DB frame shape
    save_rds(to_sql_data) # save 
   

    
    


