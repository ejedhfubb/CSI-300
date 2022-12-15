import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
proxyHost = "XXX"
proxyPort = "XXX"
proxyUser = "XXX"
proxyPass = "XXX"
proxyMeta = "http://%(user)s:%(pass)s@%(host)s:%(port)s" % {
      "host" : proxyHost,
      "port" : proxyPort,
      "user" : proxyUser,
      "pass" : proxyPass,
  }

  # 设置 http和https访问都是用HTTP代理
proxies = {
      "http"  : proxyMeta,
      "https" : proxyMeta,
  }
lst=[]
error=[]
headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'}
for page in range(1,3):
    try:
        url='http://guba.eastmoney.com/list,zssh000300_{}.html'.format(page)
        html = requests.get(url=url, headers=headers,proxies=proxies,timeout=5)
        soup=BeautifulSoup(html.text,'lxml')
        sj_lst = soup.select('.articleh.normal_post')
        for i in sj_lst:
            data={}
            data['阅读'] = i.select('span')[0].text
            data['评论'] = i.select('span')[1].text
            data['标题'] = i.select('span')[2].text
            data['作者'] = i.select('span')[3].text
            data['最后更新'] = i.select('span')[4].text
            data['页码'] = page
            href = i.select('span')[2].select('a')[0]['href']
            if 'caifuhao' in href:
                data['链接'] = 'https:'+href
            else:
                data['链接'] = 'https://guba.eastmoney.com'+i.select('span')[2].select('a')[0]['href']
            lst.append(data)
        print(page)
        time.sleep(1)
    except:
        error.append(page)
        time.sleep(1)
for x in range(5):
    error2=error.copy()
    error=[]
    for page in error2:
        try:
            url='http://guba.eastmoney.com/list,zssh000300_{}.html'.format(page)
            html = requests.get(url=url, headers=headers,proxies=proxies,timeout=5)
            soup=BeautifulSoup(html.text,'lxml')
            sj_lst = soup.select('.articleh.normal_post')
            for i in sj_lst:
                data={}
                data['阅读'] = i.select('span')[0].text
                data['评论'] = i.select('span')[1].text
                data['标题'] = i.select('span')[2].text
                data['作者'] = i.select('span')[3].text
                data['最后更新'] = i.select('span')[4].text
                data['页码'] = page
                href = i.select('span')[2].select('a')[0]['href']
                if 'caifuhao' in href:
                    data['链接'] = 'https:'+href
                else:
                    data['链接'] = 'https://guba.eastmoney.com'+i.select('span')[2].select('a')[0]['href']
                lst.append(data)
            print(page)
            time.sleep(1)
        except:
            error.append(page)
            time.sleep(1)

result = pd.DataFrame(lst)
result['最后更新'] = result['最后更新'].astype('str')
result.to_excel('result.xlsx',index=None)
