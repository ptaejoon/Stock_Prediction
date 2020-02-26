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
    predict('2020-01-02')
