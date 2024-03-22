#!/usr/bin/env python
# coding: utf-8

# ## Data Wrangling

# In[1]:


# importing libraries
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf


# In[2]:


df_symbols = pd.read_csv('stock_market_data/symbols_valid_meta.csv')


# In[3]:


df_symbols.head()


# In[4]:


df_symbols.info()


# In[5]:


assert len(df_symbols) == df_symbols['Symbol'].nunique()
num_symbols_in_meta_csv = len(df_symbols)
print('number of stocks:',num_symbols_in_meta_csv)


# In[6]:


num_in_stock_folder = 5884 # counted from folder
num_in_etf_folder = 2165

# checking that all stocks and etfs are accounted for in my dataset
num_in_stock_folder + num_in_etf_folder == num_symbols_in_meta_csv


# ## Examining single stock structure

# In[7]:


apple = pd.read_csv('stock_market_data/stocks/AAPL.csv')
apple.tail(3)


# In[8]:


apple.info()


# In[9]:


# Creating pandas_ta strategy to generate more advanced metrics

MyStrategy = ta.Strategy(
    name="SMAs, EMAs, rsi, BBs, and MACD",
    ta=[
        {"kind": "sma", "length": 20}, # Simple Moving Average
        {"kind": "sma", "length": 50},
        {"kind": "sma", "length": 200},
        {"kind": "ema", "length": 20},  # Exponential Moving Average
        {"kind": "rsi"}, # Relative Strength Index - indicator of momentum
        {"kind": "bbands", "length": 20, "col_names": ("BBL", "BBM", "BBU", "BBB", "BBP")},
        {"kind": "macd", "fast": 8, "slow": 21, "col_names": ("MACD", "MACD_H", "MACD_S")}
    ]
)


# In[10]:


def formatter(symbol):
    '''
    takes stock symbol as string
    imports data into pandas df, converts date column to datetime object, sets date as the index
    uses pandas_ta to generate new metrics for each dataframe,
    '''
    
    # reading data from csv file
    df = pd.read_csv('stock_market_data/stocks/' + symbol + '.csv')

    # converting Date column values to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])

    # setting index to be our datetime objects (pandas_ta requires a datetime index)
    df.set_index('Date', inplace = True)
    
    # generating new metrics and adding them to DataFrame
    df.ta.strategy(MyStrategy)

    return df


# In[11]:


def date_selector(df, start_year, end_year):
    '''
    takes dataframe and slices it for our start and end years (end year not inclusive)
    start and end years are taken as integers
    '''

    # slicing df
    df = df[str(start_year)+'-01-01' : str(end_year)+'-01-01']
    
    # checks for missing values
    if all(df.isna().sum() == 0):
        return df
    else:
        print('Desired window is missing values')


# In[12]:


#checking that functions work
amazon = formatter('AMZN')
amazon_2010_2020 = date_selector(amazon,2010,2020)
amazon_2010_2020.head(3)


# In[13]:


def plot_sma(symbol, start_year, end_year):
    '''
    takes stock symbol and integer start and end years
    plots that stock over that time window along with the simple moving averages for 20, 50, and 200 days
    '''
    # generating dataframe
    df = formatter(symbol)
    df = date_selector(df,start_year,end_year)
    
    plt.figure(figsize=(15,9))
    
    # manually 
    # df['Adj Close'].loc['2014-01-01':'2019-12-31'].rolling(window=20).mean().plot(label='20 Day Avg')
    # df['Adj Close'].loc['2014-01-01':'2019-12-31'].rolling(window=50).mean().plot(label='50 Day Avg')
    # df['Adj Close'].loc['2014-01-01':'2019-12-31'].rolling(window=200).mean().plot(label='200 Day Avg')

    df['SMA_20'].plot(label='20 Day SMA')
    df['SMA_50'].plot(label='50 Day SMA')
    df['SMA_200'].plot(label='200 Day SMA')
    df['Close'].plot(label=f"{symbol} stock")

    plt.grid(True)
    plt.title('20, 50, and 200 Day SMAs for ' + symbol + ' stock', color = 'black', fontsize = 20)
    plt.xlabel('Date', color = 'black', fontsize = 15)
    plt.ylabel('Stock Price (close)', color = 'black', fontsize = 15);
    plt.legend()


# In[14]:


plot_sma('MSFT',2007,2010)
# we can see the moving averages with smaller windows fit much more closely to the stocks value


# In[15]:


def plot_ema(symbol, start_year, end_year):
    '''
    Exponential Moving Average - more recent values are weighted more heavily within time window
    takes stock symbol and integer start and end years
    plots that stock over that time window along with the 20 day exponential moving average
    '''
    # generating dataframe
    df = formatter(symbol)
    df = date_selector(df,start_year,end_year)
    
    plt.figure(figsize=(15,9))

    df['EMA_20'].plot(label='20 Day EMA')
    df['Close'].plot(label=f"{symbol} stock")

    plt.grid(True)
    plt.title('20 Day EMA for ' + symbol + ' stock', color = 'black', fontsize = 20)
    plt.xlabel('Date', color = 'black', fontsize = 15)
    plt.ylabel('Stock Price (close)', color = 'black', fontsize = 15);
    plt.legend()


# In[16]:


plot_ema('MSFT',2007,2010)


# In[17]:


def plot_macd(symbol, start_year, end_year):
    '''
    -The MACD line is calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA.
    -The signal line is a nine-period EMA of the MACD line.
    source: https://www.investopedia.com/terms/m/macd.asp
    
    takes stock symbol and integer start and end years
    plots that stock over that time window along with the MACD
    '''
    # generating dataframe
    df = formatter(symbol)
    df = date_selector(df,start_year,end_year)
    
    plt.figure(figsize=(15,9))

    df['MACD'].plot(label='MACD')
    df['MACD_S'].plot(label='Signal')
    #df['MACD_H'].plot(label='MACD_H') - this is just the difference between macd and the signal so it's not helpful
    # see pandas_ta documentation: https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/momentum/macd.py

    plt.grid(True)
    plt.title('MACD for ' + symbol + ' stock', color = 'black', fontsize = 20)
    plt.xlabel('Date', color = 'black', fontsize = 15)
    plt.legend()


# In[18]:


plot_macd('MSFT',2007,2008)


# In[19]:


def plot_rsi(symbol, start_year, end_year):
    '''
    -RSI is a momentum indicator between 0 and 100. 
    -An RSI below 30 indicates undervaluation and that it may be a good time to buy.
    -An RSI over 70 indicates overvaluation and that it may be a good time to sell. - this could be useful in the end of this project
    -Most commonly calculated over a 14 day timeframe given pandas_ta using RSI_14 as default length
    Source: https://www.investopedia.com/articles/active-trading/042114/overbought-or-oversold-use-relative-strength-index-find-out.asp
    
    takes stock symbol and integer start and end years
    plots RSI for that time window
    '''
    # generating dataframe
    df = formatter(symbol)
    df = date_selector(df,start_year,end_year)
    
    plt.figure(figsize=(15,9))

    df['RSI_14'].plot(label='RSI')
    
    plt.grid(True)
    plt.axhline(0, linestyle='--', alpha = 0.5, color='gray')
    plt.axhline(30, linestyle='--', alpha = 0.5, color='green', label='Buy Signal') # line signalling buy
    plt.axhline(70, linestyle='--', alpha = 0.5, color='red', label='Sell Signal') # line signalling sell
    plt.axhline(100, linestyle='--', alpha = 0.5, color='gray')
    plt.title('RSI for ' + symbol + ' stock', color = 'black', fontsize = 20)
    plt.xlabel('Date', color = 'black', fontsize = 15)
    plt.legend()


# In[20]:


plot_rsi('MSFT',2007,2008)
# there's only one point in 2007 where our RSI would have indicated buy - in mid-March
# two moments where our RSI would have indicated sell - May and October/November


# In[21]:


def plot_bbands(symbol, start_year, end_year):
    '''
    -Bollinger Bands are a popular volatility(how quickly prices change over a period of time) indicator
    -The middle line is the 20 Day SMA
    -Upper and lower bounds are 2 standard deviations away from the SMA
    
    Source: https://www.investopedia.com/articles/active-trading/042114/overbought-or-oversold-use-relative-strength-index-find-out.asp


    Our BBand columns are
    BBL - lower
    BBM - middle - Simple moving average
    BBU - upper
    BBB - bandwidth
    BBP - percent
    takes stock symbol and integer start and end years
    plots Bbands upper middle and lwoer for that time window
    '''
    # generating dataframe
    df = formatter(symbol)
    df = date_selector(df,start_year,end_year)
    
    plt.figure(figsize=(15,9))

    df['Close'].plot(label='Close Price')
    df['BBM'].plot(label='Simple Moving Average 20 day')
    #df['BBL'].plot(color='black',label='Lower Band')
    #df['BBU'].plot(color='black',label='Upper Band')
    
    plt.grid(True)
    plt.fill_between(df.index, df['BBU'], df['BBL'], color='lightgrey', label='bollinger bandwidth') # showing only the area that the bounds envelope
    plt.title('Bollinger Bands for ' + symbol + ' stock', color = 'black', fontsize = 20)
    plt.xlabel('Date', color = 'black', fontsize = 15)
    plt.legend()


# In[22]:


plot_bbands('MSFT',2008,2010)


# In[23]:


# Autocorrelation - stocks are very mean reverting - have negative autocorrelation

microsoft = formatter('MSFT')
microsoft_2008_2010 = date_selector(microsoft,2008,2010)
microsoft_fd = microsoft_2008_2010.diff().dropna() # first difference to make data stationary

# Plot the acf function
plot_acf(microsoft_fd['Close'], lags=20, alpha=0.05)
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function of Microsoft Stock Closing Prices')
plt.show()


# In[24]:


# indicates we can indeed forecast our data from the past because of the significant non zero autocorrelation at lag 1


# In[ ]:




