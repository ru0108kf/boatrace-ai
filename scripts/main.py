import os
from datetime import datetime
from scraper import BoatraceScraper
from analyzer import BoatraceAnalyzer

# 今日の日付を取得し、YYYY-MM-DD形式に変換
today_date = datetime.now().strftime('%Y-%m-%d')

# 共通設定
folder = "C:\\Users\\msy-t\\boatrace-ai\\data"
B_url = "https://www1.mbrace.or.jp/od2/B/dindex.html"
K_url = "https://www1.mbrace.or.jp/od2/K/dindex.html"
today_date = "2025-03-30"#datetime.now().strftime('%Y-%m-%d')

scraper = BoatraceScraper(folder, BorK="K")
current_date = scraper.find_current_dates(today_date=today_date,BorK="K")
if current_date!=None:
    if current_date==today_date:
        scraper.scrape_and_process_data_for_single_day(target_date=today_date)
    else:
        scraper.scrape_and_process_data_for_date_range(start_date=current_date,end_date=today_date)

analyzer = BoatraceAnalyzer(folder = "C:\\Users\\msy-t\\boatrace-ai\\data")

print("選手登録番号を入力してください。")
boat_number_1 = int(input("1号艇選手登録番号:"))
boat_number_2 = int(input("2号艇選手登録番号:"))
boat_number_3 = int(input("3号艇選手登録番号:"))
boat_number_4 = int(input("4号艇選手登録番号:"))
boat_number_5 = int(input("5号艇選手登録番号:"))
boat_number_6 = int(input("6号艇選手登録番号:"))

outputs = analyzer.get_boatrace_data(start_date="2024-11-01", end_date="2025-03-30", venue="全国",
                            boat_number_1=boat_number_1, boat_number_2=boat_number_2,
                            boat_number_3=boat_number_3, boat_number_4=boat_number_4,
                            boat_number_5=boat_number_5, boat_number_6=boat_number_6)
print(outputs)