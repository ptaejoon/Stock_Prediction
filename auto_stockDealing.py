import schedule
import time
from datetime import date
def execute_files():
    exec(open("crawler/CrawlingOneDay.py").read())
    exec(open("udf/GAN_Predictor.py").read())
    print(date.today(),"just finished prediction")
def execute():
    exec(open("test.py").read())
#schedule.every().day.at("23:30").do(execute_files)
schedule.every().day.at("14:07").do(execute)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
