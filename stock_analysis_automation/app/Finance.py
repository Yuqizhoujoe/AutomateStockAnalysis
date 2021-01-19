import yfinance as yf, shutil, os, time, glob, smtplib, ssl
from get_all_tickers import get_tickers as gt
import time
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pretty_html_table as html_table

# Global Variable
MARKET_CAP_MAX = 5000000
MARKET_CAP_MIN = 50000
API_CALLS_MAX = 1000
PATH = '/Graduate/Python/stock_investment/stock_analysis_automation/app/data/'

# Class
class Stock:
     def __init__(self):
         self.getTickers()
         self.data = self.getStockInfo()

     def getTickers(self):
         self.tickers = gt.get_tickers_filtered(mktcap_min=MARKET_CAP_MIN, mktcap_max=MARKET_CAP_MAX)
         print('The number of tickers: %s' % (str(len(self.tickers))))

     def getStockInfo(self):
         amount_of_API_calls = 0
         stock_failure = 0
         stock_not_imported = 0

         i = 0
         temp = {
             'ticker': [],
             'gain': [],
             'obv': []
         }
         while((i < len(self.tickers)) and (amount_of_API_calls < API_CALLS_MAX)):
             try:
                 stock = self.tickers[i]
                 data = yf.Ticker(str(stock)).history(period="1mo")
                 close = data['Close'].tolist()
                 gain = 0
                 if close and len(close) > 0:
                     recentPrice = close[0]
                     priorPrice = close[len(close)-1]
                     if recentPrice and priorPrice:
                         gain = recentPrice/priorPrice - 1
                         gain = round(gain, 4)
                 obv = self.getOBV(data)
                 temp['ticker'].append(stock)
                 temp['gain'].append(gain)
                 temp['obv'].append(obv)
                 print("#%d - Ticker: %s; Gain: %.2f; OBV: %.2f" % (i, stock, gain, obv))
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

         print("The number of stocks imported: " + str(i - stock_not_imported))
         df = pd.DataFrame(data=temp)
         df.dropna(subset=['obv'], inplace=True)
         df['Performance(gain)'] = df['gain'].rank(ascending=False, na_option="top").astype(int)
         df['Performance(obv)'] = df['obv'].rank(ascending=False, na_option="top").astype(int)
         df.sort_values(['gain', 'obv'], inplace=True, ascending=False)
         return df

     def getOBV(self, df):
         pos_move = []
         neg_move = []
         obv = 0
         count = 0
         while (count < len(df)):
             close = df['Close'][count]
             open = df['Open'][count]
             volume = df['Volume'][count]
             if volume and close and open:
                 normalizedVol = volume / open
                 if (close > open):
                     pos_move.append(count)
                     obv += normalizedVol
                 else:
                     neg_move.append(count)
                     obv -= normalizedVol
             count += 1
         return obv

     def get(self):
         return self.data
class Email:
     EMAIL_FROM = "joepicksstockforyou@gmail.com"
     SMTP_PASSWORD = "yuqizhou1234"
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

# Implementation
stock = Stock()
df = stock.get()
top = html_table.build_table(df.head(20), "yellow_dark")
bottom = html_table.build_table(df.tail(20), "orange dark")
email = Email()
email.send_email(top, bottom)
df.to_csv(PATH + "Stock Report.csv")