import stockGAN
import datetime
def predict(days):
    #days : datetime
    gan = stockGAN.GAN(batch_size=40)
    #gan.train(1)
    #gan.load(num=0)
    time = ' 15:30:00'
    days_to_datetime = datetime.datetime.strptime(days+time,'%Y-%m-%d %H:%M:%S')
    print(gan.predict(days_to_datetime))

if __name__ == '__main__':
    predict_day = input("주식 가격을 예측하고자 하는 날의 날짜를 입력하시오. (param. 0000-00-00)")
    predict('2020-01-02')
