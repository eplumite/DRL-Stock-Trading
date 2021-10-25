import datetime
import apimoex
import numpy as np
import pandas as pd
import requests

import os

def download_data():

    folder_with_data = "data"
    if not os.path.exists(folder_with_data):
        os.mkdir("data")

    if not os.path.exists("data/hour"):
        os.mkdir("data/hour")
    if not os.path.exists("data/day"):
        os.mkdir("data/day")
    if not os.path.exists("data/week"):
        os.mkdir("data/week")

    start = datetime.datetime(2005, 1, 1, 0, 0)
    end = datetime.datetime(2016, 1, 1, 0, 0)

    # IMOEX: RU000A0JP7K5
    arguments = None
    request_url = (
            "https://iss.moex.com/iss/statistics/engines/stock/markets/index/analytics/IMOEX/tickers.json"
        )  

    with requests.Session() as session:
        iss_data = apimoex.ISSClient(session, request_url, arguments).get()
        imoex_tickers_df = pd.DataFrame(iss_data['tickers'])

        imoex_tickers_df.drop(columns=['tradingsession'], inplace=True)
        imoex_tickers_df['from'] = pd.to_datetime(imoex_tickers_df['from'], format='%Y-%m-%d')
        imoex_tickers_df['till'] = pd.to_datetime(imoex_tickers_df['till'], format='%Y-%m-%d')

        imoex_tickers_df = imoex_tickers_df[(imoex_tickers_df['from'] < start) &
                                            (imoex_tickers_df['till'] > end)]
        imoex_tickers_df = imoex_tickers_df.reset_index(drop=True)
        imoex_tickers_df.to_csv('data/tickers.csv', index=False)

        tickers = imoex_tickers_df['ticker']
        for ticker in tickers:
            ticker_data = apimoex.get_market_candles(session, ticker, interval=60, 
                                                     start='2005-01-01', end='2016-01-01',
                                                     columns=('begin','open','close','high','low'))
            ticker_data_df = pd.DataFrame(ticker_data).sort_values(by=['begin']).reset_index(drop=True)
            ticker_data_df.to_csv(f'data/hour/{ticker}.csv', index=False)

            ticker_data = apimoex.get_market_candles(session, ticker, interval=24, 
                                                     start='2005-01-01', end='2016-01-01',
                                                     columns=('begin','open','close','high','low'))
            ticker_data_df = pd.DataFrame(ticker_data).sort_values(by=['begin']).reset_index(drop=True)
            ticker_data_df.to_csv(f'data/day/{ticker}.csv', index=False)

            ticker_data = apimoex.get_market_candles(session, ticker, interval=7, 
                                                     start='2005-01-01', end='2016-01-01',
                                                     columns=('begin','open','close','high','low'))
            ticker_data_df = pd.DataFrame(ticker_data).sort_values(by=['begin']).reset_index(drop=True)
            ticker_data_df.to_csv(f'data/week/{ticker}.csv', index=False)


def load_historical_data():
    
    columns = ["timestamp"] + list(pd.read_csv("data/tickers.csv")["ticker"])
    historical_data = pd.DataFrame(columns=columns)

    historical_data["timestamp"] = pd.read_csv(f"data/day/{columns[1]}.csv")['begin']
    
    for ticker in columns[1:]:
        ticker_data = pd.read_csv(f"data/day/{ticker}.csv")
        historical_data[ticker] = ticker_data['open']
        
    historical_data.fillna(np.inf, inplace=True)

    return historical_data