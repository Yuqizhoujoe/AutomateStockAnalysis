import yfinance as yf, shutil, os, time, glob, smtplib, ssl
import time
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import requests
import re
from bs4 import BeautifulSoup
import praw
import datetime
import numpy as np

# Global Variable
MILLION = 1000000
MARKET_CAP_MIN = 5000 * MILLION * 2
API_CALLS_MAX = 1000
PATH = 'PATH'
API_KEY = 'API_CODE'
MAIN_API = f'https://financialmodelingprep.com/api/v3/'
INSIDER_API = f'https://financialmodelingprep.com/api/v4/'
KEY_URL = f'apikey={API_KEY}'
TICKER_URL = MAIN_API + f'stock-screener?marketCapMoreThan=0&' + KEY_URL
REDDIT_CLIENT_ID = 'REDDIT_CLIENT_ID'
REDDIT_CLIENT_SECRET = 'REDDIT_CLIENT_SECRET'
REDDIT_USER_AGENT = 'Reddit WebScrapping'
redditUrl = 'https://www.reddit.com/r/stocks/comments/lqgs3t/rstocks_daily_discussion_technicals_tuesday_feb/'

# Class
class Stock:
     def __init__(self, tickers):
         self.tickers = tickers
         self.data = self.getStockInfo()

     def getStockInfo(self):
         amount_of_API_calls = 0
         stock_failure = 0
         stock_not_imported = 0

         i = 0
         temp = {
             'ticker': [],
             'weekly%': [],
             'monthly%': [],
             'yearly%': [],
             '$': [],
             'obv': [],
             'acquisition': [],
             'disposition': [],
             'totalInsiderTrading': [],
             'sector': [],
             'ipoDate': [],
             'rsi': [],
             'eps': [],
             'pe': []
         }
         while(i < len(self.tickers)):
             try:
                 stock = self.tickers[i]
                 today = datetime.datetime.today().date()
                 yearAgo = (datetime.datetime.today() - datetime.timedelta(12 * 365 / 12)).date()
                 url = MAIN_API + f'historical-price-full/{stock}?from={yearAgo}&to={today}&' + KEY_URL
                 print(url)
                 data = requests.get(url).json()
                 stockData = []
                 if data:
                    stockData = data['historical']
                 currentPrice = weeklyReturn = monthlyReturn = 0
                 if len(stockData) > 0:
                    currentPrice = stockData[0]['close']
                    yearlyReturn = currentPrice / stockData[len(stockData)-1]['close'] - 1
                 if len(stockData) >= 5:
                     weeklyReturn = currentPrice / stockData[4]['close'] - 1
                 if (len(stockData) >= 21):
                     monthlyReturn = currentPrice / stockData[20]['close'] - 1
                 obv = 0
                 if len(stockData) >= 21:
                    obv = self.getOBV(stockData, stock)
                 rsi = 0
                 if len(stockData) >= 28:
                    rsi = self.getRsi(stockData)
                 ipoDate, sector = self.getProfile(stock)
                 pe, eps = self.getQuote(stock)
                 acquisition, disposition, totalInsiderTrading = self.getInsiderTrading(stock)
                 temp['ticker'].append(stock)
                 temp['weekly%'].append(weeklyReturn)
                 temp['monthly%'].append(monthlyReturn)
                 temp['yearly%'].append(yearlyReturn)
                 temp['$'].append(currentPrice)
                 temp['obv'].append(obv)
                 temp['acquisition'].append(acquisition)
                 temp['disposition'].append(disposition)
                 temp['totalInsiderTrading'].append(totalInsiderTrading)
                 temp['rsi'].append(rsi)
                 temp['sector'].append(sector)
                 temp['ipoDate'].append(ipoDate)
                 temp['eps'].append(eps)
                 temp['pe'].append(pe)
                 print("#%d - Ticker: "
                       "%s; Current $: %.2f; Weekly: %.2f; OBV: %.2f; TotalInsiderTrading: %d; RSI: %d; Sector: %s; EPS: %s; PE: %s"
                       % (i, stock, currentPrice, weeklyReturn, obv, totalInsiderTrading, rsi, sector, eps, pe))
                 time.sleep(2)
                 amount_of_API_calls += 1
                 stock_failure = 0
                 i += 1
             except ValueError:
                 print("Server ERROR, attempt to fix")
                 if stock_failure > 5:
                     i += 1
                     stock_not_imported += 1
                 amount_of_API_calls += 1
                 stock_failure += 1

         print("The number of stocks imported: " + str(len(stockData)))
         df = pd.DataFrame(data=temp)
         df['Performance(weekly%)'] = df['weekly%'].rank(ascending=False, na_option="top").astype(int)
         df['Performance(obv)'] = df['obv'].rank(ascending=False, na_option="top").astype(int)
         df.sort_values(['weekly%', 'obv'], inplace=True, ascending=False)
         return df

     def getOBV(self, df, stock):
         obv = 0
         if df and stock != 'FLAT':
             pos_move = []
             neg_move = []
             count = 21
             while (count >= 0):
                 close = df[count]['close']
                 open = df[count]['open']
                 volume = df[count]['volume']
                 if volume and close and open:
                     normalizedVol = volume / open
                     if (close > open):
                         pos_move.append(count)
                         obv += normalizedVol
                     else:
                         neg_move.append(count)
                         obv -= normalizedVol
                 count -= 1
         return obv

     def getProfile(self, ticker):
         url = MAIN_API + f'profile/{ticker}/?' + KEY_URL
         res = requests.get(url).json()
         ipoDate = ''
         sector = ''
         if len(res) > 0:
            data = res[0]
            ipoDate = data['ipoDate']
            sector = data['sector']
         return ipoDate, sector

     def getQuote(self, ticker):
         url = MAIN_API + f'quote/{ticker}?' + KEY_URL
         res = requests.get(url).json()
         pe = 0
         eps = 0
         if res and len(res) > 0:
             data = res[0]
             pe = data["pe"]
             eps = data['eps']
         return pe, eps


     def getInsiderTrading(self, ticker):
         acquisition = disposition = totalInsiderTrading = 0
         try:
             url = INSIDER_API + f'insider-trading?symbol={ticker}&' + KEY_URL
             res = requests.get(url).json()

             for data in res:
                 if data['acquistionOrDisposition'] == 'A':
                     acquisition += 1
                     totalInsiderTrading += data['securitiesTransacted']
                 if data['acquistionOrDisposition'] == 'D':
                     disposition += 1
                     totalInsiderTrading -= data['securitiesTransacted']
         except:
            return acquisition, disposition, totalInsiderTrading
         return acquisition, disposition, totalInsiderTrading

     def getRsi(self, df):
         # Observe the last 14 closing prices of a stock
         # Determine whether the current day's closing price is higher or lower than prev day
         # Calculate the average gain and loss over the last 14 days
         # Compute the relative strength (RS) Average Gain / Average Loss
         # Compute the relative strength Index (RSI) 100 - 100 / (1 + RS)
         closes = []
         i = 27
         while i >= 0:
             if df[i]['close'] > float(2.00):
                 closes.append(df[i]['close'])
             i -= 1
         i = 0
         upPirces = []
         downPrices = []
         while i < len(closes):
             if i == 0:
                 upPirces.append(0)
                 downPrices.append(0)
             else:
                 diff = closes[i] - closes[i - 1]
                 if (diff > 0):
                     upPirces.append(diff)
                     downPrices.append(0)
                 else:
                     downPrices.append(diff)
                     upPirces.append(0)
             i += 1
         x = 0
         avgGain = []
         avgLoss = []
         while x < len(upPirces):
             if x < 15:
                 avgGain.append(0)
                 avgLoss.append(0)
             else:
                 sumGain = 0
                 sumLoss = 0
                 y = x - 14
                 while y <= x:
                     sumGain += upPirces[y]
                     sumLoss += downPrices[y]
                     y += 1
                 avgGain.append(sumGain / 14)
                 avgLoss.append(abs(sumLoss / 14))
             x += 1
         p = 0
         RS = []
         RSI = []
         while p < len(closes):
             if p < 15:
                 RS.append(0)
                 RSI.append(0)
             else:
                 RSValue = 0
                 if avgLoss[p] != 0:
                    RSValue = (avgGain[p] / avgLoss[p])
                 RS.append(RSValue)
                 rsi = 0
                 if (1 + RSValue != 0):
                     rsi = 100 - (100 / (1 + RSValue))
                 RSI.append(rsi)
             p += 1
         currRSI = 0
         if (len(RSI) >= 16):
             currRSI = RSI[15]
         return currRSI

     def get(self):
         return self.data
class Email:
     EMAIL_FROM = "Email Address"
     SMTP_PASSWORD = "Password"
     BODY_MESSAGE = """
         <div>
             <div>
                 <h1> Please check the stock report below </h1>
                 <ol>
                     <li><em>gain - stock recent monthly return </em></li>
                     <li><em>obv - On-Balance Value</em></li>
                 </ol>
             </div>
             <br> 
             <br>
             <h3 style="color: #DE5F44"> If you do not understand some of the terms and how do I process the data, either google/baidu or reach me out on Wechat</h4>
         </div>
         """
     msg = MIMEMultipart('mixed')
     msg['Subject'] = "Joe Joe Picks Stock For You"
     msg['From'] = EMAIL_FROM

     def __init__(self, address = "kevinyuqi123@gmail.com"):
         self.address = address

     def send_email(self, top, bottom, address=None):
         if address:
             self.msg['To'] = address
         else:
             self.msg['To'] = self.address

         MESSAGE_BODY = MIMEMultipart('alternative')
         MESSAGE_BODY.attach(MIMEText(
             (
                     """\
                     <html>
                         <body>
                             <div>
                                 """ + self.BODY_MESSAGE + """
                                 <br>
                                 <h3> Top 20 Stocks </h3>
                                 """ + top + """
                                 <hr>
                                 <h3> Bottom 20 Stocks </h3>
                                 """ + bottom + """
                                 <hr>
                             </div> 
                         </body>
                     </html>
                     """
             ), "html"
         ))
         self.msg.attach(MESSAGE_BODY)

         # Create SMTP object
         with smtplib.SMTP("smtp.gmail.com", 587) as server:
             server.ehlo()
             server.starttls()
             server.ehlo()
             # Login to the server
             server.login(self.EMAIL_FROM, self.SMTP_PASSWORD)
             # Convert the message to a string and send it
             server.sendmail(self.msg['From'], self.msg['To'], self.msg.as_string())
             server.quit()
class Reddit:
    def __init__(self):
        self.tickersFromReddit = self.getTickers()

    def scrapeFromReddit(self, allTickers):
        self.reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)
        reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET,
                             user_agent=REDDIT_USER_AGENT)
        submission = reddit.submission(
            url=redditUrl)
        submission.comments.replace_more(limit=0)
        tickers = []
        for comment in submission.comments.list():
            text = re.sub('[^a-zA-Z]+', ' ', comment.body)
            text = text.upper()
            text = text.strip()
            for word in text.split(" "):
                if word in allTickers and word not in tickers:
                    tickers.append(word)
        return tickers

    def getTickers(self):
        tickers = []
        allTickers = requests.get(TICKER_URL).json()
        for ticker in allTickers:
            tickers.append(ticker['symbol'])
        print('The number of tickers: %s' % (str(len(tickers))))
        tickersFromReddit = self.scrapeFromReddit(tickers)
        print('Get %s stock tickers from Reddit r/stocks' % (str(len(tickersFromReddit))))
        return tickersFromReddit

    def reddit(self, tickers, url):
        # url
        url = url
        # create a empty list
        data = []
        # access the webpage as Chrome
        my_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'}

        # initialize src False
        src = False

        # try to scrape 5 times
        for i in range(1, 6):
            try:
                # get url content
                response = requests.get(url, headers=my_headers)
                # get html content
                src = response.content
                break
            except:
                print('failed attempt #', i)
                # wait 2 secs before next time
                time.sleep(2)

        # if we could not get the page
        if not src:
            print('Could not get page', url)
        else:
            print('Successfully get the page', url)

        soup = BeautifulSoup(src.decode('ascii', 'ignore'), 'lxml')
        div = soup.find('div', {'class': re.compile('_1ump7uMrSA43cqok14tPrG')});
        textDivs = div.findAll('div', {'class': re.compile('RichTextJSON-root')});
        for textDiv in textDivs:
            textArray = textDiv.findAll('p')
            for elem in textArray:
                text = elem.text.strip()
                words = text.split(" ");
                for word in words:
                    if word in tickers and word not in data:
                        data.append(word.upper())
        return data

    def get(self):
        return self.tickersFromReddit
def readRobinHoodHoldingFromTxt():
    robinHood = []
    with open(PATH + 'MyHolding.txt', 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            line = line.strip()
            robinHood.append(line)
    return robinHood
def readArkEfts():
    files = (glob.glob(PATH + 'ARK*.csv'))
    tickers = []
    for file in files:
        df = pd.read_csv(file)
        list = df['ticker'].tolist()
        tickers.extend(x for x in list if x not in tickers and str(x).isnumeric() is False)
    return tickers

# Implementation
reddit = Reddit()
tickersFromReddit = reddit.get()
tickersFromRobinHood = readRobinHoodHoldingFromTxt()
tickersFromARKEtfs = readArkEfts()
tickers = list(tickersFromRobinHood)
tickers.extend(x for x in tickersFromARKEtfs if x not in tickersFromRobinHood)
tickers.extend(x for x in tickersFromReddit if x not in tickersFromRobinHood)
stock = Stock(tickers)
df = stock.get()
top = html_table.build_table(df.head(20), "yellow_dark")
bottom = html_table.build_table(df.tail(20), "orange_dark")
email = Email()
email.send_email(top, bottom)
date = datetime.datetime.today().date()
df.to_csv(PATH + f'Stock Report {date}.csv')
