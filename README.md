# Forex-Recovery
A model created based on LSTM, channel support and resistance and economic calendar for predicting the recovery rate in Forex market.

Considering the current situation of EURO and British Pounds had fallen by **13.45%** and **17.33%** YTD wrt to the US Dollar respectevly whereas GBPUSD is **2.34%** behind comparing with its all-time low.
There are different reasons for that but excluding the intuition and geo-political situation we can dig into data analytics to find what could be the recovery rate

# Model Arch
![Model](https://github.com/white07S/Forex-Recovery/blob/main/model/model.png)



Alright, so i still see few mistakes 
let me give me a summary of this work again.
model1: for predicting the parameter attributes of technical indicators(because these params changes when time interval changes of OHLCV)(
like (timeperiod, nbdevup, nbdevdn) in Bollinger bands case, three differnet time period in SMA,in Stoch we predict these (fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype) and in ATR case timeperiod) and this time you even forgot to use technical indicator in the strategy.
model2: Q learning based on reward and risk, using it to predict the best possible stop loss and take profit. (an example: pass the trade signal with datetime,price) and it gives you SL and TP
model3: Anamoly detection: so when finally trade signal is being produced by the "Strategy" we want to verify with anomoly detection. (for ex: pass trade signal, OHLC data and support and resistance  and then its predicts if its anomaly or not )

now strategy:

step 1: collect technical indictor: we have three sma with differnet windows 3,8,13, Stochastic, ATR, Bollinger's Band (here we use model 1 to predict the parameter attributes of these technical indicators)
step 2: we do some calculation: first find pivots points and then calculate support and resistance.
step 3: logic of strategy: we buy either sma with 3 window crossover sma window with 8 or if sma(8) crossover sma(13) or we buy when its oversold, 
we use support and resistnace for confirmation but with a "or" clause. and opposite for selling 
step 4: we use model 2 to find take profit and stop loss and pass the trade signal from anamoly detection of model 3.
step 5: we punish the Q-training model if there was loss trade beacuse of stop loss hit in the trade.

I hope things are clear enough now so now 
you give me two files 
1. models.py with all the models and its training method implemented.
2. Optimized strategy with numba with all models used properly 

and both file follow Object Oriented pattern.

Let me know if something is not clear before you start messing up with things.
