# Stock_Prediction
Code about GAN is in stockGAN_v*.<br> All files outside folders are for testing.<br> Codes about Crawling and Database are in Crawling folder. 

v1:<br>
First Model in project.<br>
Input G: Averaged PV and Close,Open,Low,High,Volume of 169 KOSPI200 companies<br>
Input D: Predicted value of Close,Open,Low,High, Volume of a day and previous 10 days stock data.<br>
Result : Using Volume as D's input created massive error.<br>

v2:<br>
Drop Volume in D's input.<br>
Input G: Averaged PV and preprocessed Close,Open,Low,High,Volume of 169 KOSPI200 companies.<br>
High,Low,Min,Max has changed to percentile comparing Close price of last day.<br>
Volume has changed to certain value based on experiment. You can find the method at 'stockGAN_v2/udf_scaler.py'.<br>
Result : GAN model is still running. <br>
Lowest Average Error on Validation Data till now : 1.2 %<br>

v3:<br>
Simplified Version of Stock Prediction GAN.<br>
Drop Paragraph Vector in Generator's Input. This version only use stock index to both input.<br>
Result : We Found ,at some point, the model cannot diminsh it's error anymore.<br>
Lowest Average Error on Validation Data : 1.8 %<br>

v4:<br>
Rescaled Stock Data of v2 to -1 and 1 to broaden the predictable range.<br>
Result : GAN model is still running.<br>
Lowest Average Error on Validation Data till now : 1.9 %<br>
