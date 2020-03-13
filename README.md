# Stock_Prediction
<h5> This Project has done by ptaejoon, sinunu, suwhan. We started this project to find whether the stock market really has pattern.</h5> <br>
GAN we designed is inspired by two papers:

    Deep learning for stock prediction using numerical and textual information 
    Stock Market Prediction on High-Frequency Data Using Generative Adversarial Nets


Codes about GAN are in stockGAN_v*.<br>
LSTM with 10 timesteps was set to Generator.<br> The generator predicts stock price (Open, Close, High, Low) of 11th day.<br>
We chose CNN as Discriminator.<br>
Among KOSPI 200, only 169 companies were used as input since they have record of prices from 2010.<br>

    training set : Daily price of Open,Close,High,Low,Volume of 169 companies in KOSPI200 from 2010 to 2018.
    test set : Daily price of Open,Close,High,Low,Volume of 169 companies in KOSPI200 during 2019.

v1:

    The very first model.
    Input G: Averaged PV and Close,Open,Low,High,Volume of 169 KOSPI200 companies
    Input D: Predicted value of Close,Open,Low,High, Volume of a day and previous 10 days stock data.
    Result : Using Volume as D's input created massive error.

v2:

    Drop Volume in D's input.
    Input G: Averaged PV and preprocessed Close,Open,Low,High,Volume of 169 KOSPI200 companies.
    High,Low,Min,Max has changed to percentile comparing Close price of last day.
    Volume has changed to certain value based on experiment. You can find the method at 'stockGAN_v2/udf_scaler.py'.
    Result : GAN model is still running. 
    Lowest Average Error on Validation Data (2019) 
        Average Open Price error : 0.83 %
        Average Close Price Error : 1.46 %
        Average High Price Error : 1.23 %
        Average Low Price Error : 1.13 %

    Lowest Average Error on Validation Data (2020.1~2) 
        Average Open Price error : 1.15 %
        Average Close Price Error : 1.9 %
        Average High Price Error : 1.58 %
        Average Low Price Error : 1.4 %

v3:

    Simplified Version of Stock Prediction GAN.
    Drop Paragraph Vector in Generator's Input. This version only use stock index to both input.
    Result : We Found ,at some point, the model cannot diminsh it's error anymore.
    Lowest Average Error on Validation Data : 1.8 %

v4:

    Rescaled Stock Data of v2 to -1 and 1 to broaden the predictable range.
    Result : GAN model is still running.
    Lowest Average Error on Validation Data till now : 1.9 %

v5:
    
    Returns only Close, Max price as Generator's output.
    Result : much high error rate than v2~4.
<br>
trade.py: Code for autonomous stock trade.<br>
validate.py : Code to evaluate model's proficiency. Used 2019 and 2020.1~2 data for validation. <br>
