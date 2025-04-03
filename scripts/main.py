import os
from datetime import datetime, timedelta
from scraper import BoatraceScraper
from scraper import BoatraceLatestDataScraper
from analyzer import BoatraceAnalyzer

# 今日の日付を取得し、YYYY-MM-DD形式に変換
today_date = datetime.now().strftime('%Y-%m-%d')

# 共通設定
folder = "C:\\Users\\msy-t\\boatrace-ai\\data"
B_url = "https://www1.mbrace.or.jp/od2/B/dindex.html"
K_url = "https://www1.mbrace.or.jp/od2/K/dindex.html"

today_date = "2025-03-31"#datetime.now().strftime('%Y-%m-%d')
race_no = 2
venue = "住之江"

today_datetime = datetime.strptime(today_date, '%Y-%m-%d')
yesterday = (today_datetime - timedelta(days=1)).strftime('%Y-%m-%d')


str_race_no = ''.join(chr(ord(char) + 0xFEE0) if '!' <= char <= '~' else char for char in str(race_no))

scraper = BoatraceScraper(folder, BorK="K")
current_date = scraper.find_current_dates(date=yesterday,BorK="K")
if current_date!=None:
    if current_date==today_date:
        scraper.scrape_and_process_data_for_single_day(target_date=yesterday)
    else:
        scraper.scrape_and_process_data_for_date_range(start_date=current_date,end_date=yesterday)

csraper = BoatraceScraper(folder, BorK="B")
current_date = csraper.find_current_dates(date=today_date,BorK="B")
if current_date!=None:
    if current_date==today_date:
        csraper.scrape_and_process_data_for_single_day(target_date=today_date)
    else:
        csraper.scrape_and_process_data_for_date_range(start_date=current_date,end_date=today_date)

boat_number_1, boat_number_2, boat_number_3, boat_number_4, boat_number_5, boat_number_6 = csraper.get_player_numbers_by_race(date=today_date,venue=venue,race_number=str_race_no)

analyzer = BoatraceAnalyzer(folder)

outputs = analyzer.get_boatrace_data(start_date="2024-11-01", end_date=yesterday, venue="全国",
                            boat_number_1=boat_number_1, boat_number_2=boat_number_2,
                            boat_number_3=boat_number_3, boat_number_4=boat_number_4,
                            boat_number_5=boat_number_5, boat_number_6=boat_number_6)

latestdata = BoatraceLatestDataScraper()
latestoutputs = latestdata.get_latest_boatrace_data(place_name=venue,race_no=race_no,date=today_date)

print(outputs)
print(latestoutputs)