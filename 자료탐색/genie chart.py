from bs4 import BeautifulSoup
import requests

# 2022
# 멜론차트는 헤더정보를 입력해줘야함
header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko'}
# response = requests.get('https://www.melon.com/chart/index.htm', headers=header)
#
# html = response.text
# soup = BeautifulSoup(html, 'html.parser')  # html.parser를 사용해서 soup에 넣겠다
# title = soup.find_all("div", {"class": "ellipsis rank01"})  # 노래제목
# singer = soup.find_all("div", {"class": "ellipsis rank02"})  # 가수
#
# real_title = []
# real_singer = []
#
# for i in title:
#     real_title.append(i.find('a').text)
#
# for j in singer:
#     real_singer.append(j.find('a').text)
#
# rank = 50
# for r in range(rank):
#     print(f'{r+1}위 {real_title[r]} - {real_singer[r]}')


# # 2002-2021
#
# years = [2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
# url = 'https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2002'
#
# resp = requests.get(url,headers=header).text
# soup2 = BeautifulSoup(resp, 'html.parser')
#
# title = soup2.select('#frm > table > tbody > div.wrap_song_info > span > strong > a')  # 노래제목
#
# titles = []
# for i in title:
#     titles.append(i.text)
#
# print(titles)

# 2002-2021

years = [2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
url = 'https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2002'

resp = requests.get(url,headers=header).text
soup2 = BeautifulSoup(resp, 'html.parser')

title = soup2.find_all("div", {"class": "ellipsis rank01"})  # 노래제목
singer = soup2.find_all("div", {"class": "ellipsis rank02"})  # 가수

titles = []
singers = []

for l in title:
    titles.append(l.find('a').text)

for m in singer:
    singers.append(m.find('a').text)

print(titles)





# # 2002~2021
# years = [2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
# url = 'https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate='
#
# for i in years:
#     soup = BeautifulSoup(requests.get(url+str(i),headers=header).text,'html.parser')
#     titles = soup.find_all('div',{'class':'ellipsis rank01'})
#
#     title=[]
#     for j in titles:
#         title.append(i.find('a').text)
#
#     print(title)

    #print(f'{titles}')
# 2022
url_2022 = 'https://www.melon.com/chart/index.htm'

# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2002
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2003
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2004
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2005
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2006
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2007
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2008
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2009
# https://www.melon.com/chart/age/index.htm?chartType=AG&chartGenre=KPOP&chartDate=2010
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2010
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2011
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2012
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2013
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2014
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2015
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2016
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2017
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2018
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2019
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2020
# https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate=2021
# https://www.melon.com/chart/index.htm